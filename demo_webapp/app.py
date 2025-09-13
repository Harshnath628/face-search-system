from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
import os
import sys
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename
import json

# Add parent directory to path to import face_search_package
sys.path.insert(0, str(Path(__file__).parent.parent))
from face_search_package import FaceSearchSystem, Config, ConfigPresets

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize the Face Search System with demo configuration
config = {
    'database_path': 'demo_face_data',
    'log_searches': True,
    'default_tolerance': 0.6
}
face_system = FaceSearchSystem(config)

# Global variable to store current session data
current_session = {}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page"""
    stats = face_system.get_statistics()
    return render_template('index.html', stats=stats)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and face detection"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_filename = str(uuid.uuid4()) + '_' + filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            try:
                file.save(filepath)
                
                # Detect faces in the uploaded image
                detection_result = face_system.search_faces_in_image(filepath)
                
                if detection_result['success'] and detection_result['total_faces'] > 0:
                    # Store session data
                    session_id = str(uuid.uuid4())
                    current_session[session_id] = {
                        'original_image': filepath,
                        'detected_faces': detection_result['faces'],
                        'total_faces': detection_result['total_faces']
                    }
                    
                    # Draw face boxes on the image for display (using detector directly)
                    face_locations = [face['location'] for face in detection_result['faces']]
                    annotated_image = face_system.detector.draw_face_boxes(filepath, face_locations)
                    
                    return render_template('face_selection.html', 
                                         session_id=session_id,
                                         detection_result=detection_result,
                                         annotated_image=annotated_image.replace('\\', '/'),
                                         original_image=filepath.replace('\\', '/'))
                else:
                    if detection_result['success']:
                        flash('No faces detected in the image')
                    else:
                        flash(f'Error detecting faces: {detection_result["error"]}')
                    return redirect(url_for('upload_file'))
                    
            except Exception as e:
                flash(f'Error processing image: {str(e)}')
                return redirect(url_for('upload_file'))
        else:
            flash('Invalid file type. Please upload an image file.')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/search/<session_id>/<int:face_id>')
def search_face(session_id, face_id):
    """Search for a specific face"""
    if session_id not in current_session:
        flash('Session expired. Please upload a new image.')
        return redirect(url_for('upload_file'))
    
    session_data = current_session[session_id]
    detected_faces = session_data['detected_faces']
    
    # Perform search using the face_id
    selected_face = None
    for face in detected_faces:
        if face['face_id'] == face_id:
            selected_face = face
            break
    
    if selected_face:
        search_result = face_system.get_matches(selected_face['encoding'])
    else:
        search_result = {'success': False, 'error': 'Face not found'}
    
    # Selected face is already found above
    
    return render_template('search_results.html',
                         search_result=search_result,
                         selected_face=selected_face,
                         face_id=face_id,
                         session_id=session_id)

@app.route('/add_profile')
def add_profile_form():
    """Show form to add new profile"""
    return render_template('add_profile.html')

@app.route('/add_profile', methods=['POST'])
def add_profile():
    """Add a new profile to the database"""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('add_profile_form'))
    
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        flash('Please select a valid image file')
        return redirect(url_for('add_profile_form'))
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    unique_filename = str(uuid.uuid4()) + '_profile_' + filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    try:
        # Add profile using the face system
        profile_id = face_system.add_profile(
            name=request.form.get('name', ''),
            image_path=filepath,
            age=request.form.get('age', ''),
            description=request.form.get('description', ''),
            metadata={
                'uploaded_by': 'web_interface',
                'original_filename': filename
            }
        )
        
        
        if profile_id:
            flash(f'Profile added successfully! ID: {profile_id}')
            return redirect(url_for('view_profiles'))
        else:
            flash('Error adding profile to database')
            return redirect(url_for('add_profile_form'))
            
    except Exception as e:
        flash(f'Error processing profile: {str(e)}')
        if os.path.exists(filepath):
            os.remove(filepath)
        return redirect(url_for('add_profile_form'))

@app.route('/profiles')
def view_profiles():
    """View all profiles in the database"""
    profiles = face_system.get_all_profiles()
    stats = face_system.get_statistics()
    return render_template('profiles.html', profiles=profiles, stats=stats)

@app.route('/profile/<profile_id>')
def view_profile(profile_id):
    """View a specific profile"""
    profile = face_system.get_profile(profile_id)
    if not profile:
        flash('Profile not found')
        return redirect(url_for('view_profiles'))
    
    return render_template('profile_detail.html', profile=profile)

@app.route('/delete_profile/<profile_id>')
def delete_profile(profile_id):
    """Delete a profile"""
    success = face_system.delete_profile(profile_id)
    if success:
        flash('Profile deleted successfully')
    else:
        flash('Error deleting profile')
    
    return redirect(url_for('view_profiles'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/search_face', methods=['POST'])
def api_search_face():
    """API endpoint for face search"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        face_id = data.get('face_id')
        
        if session_id not in current_session:
            return jsonify({'success': False, 'error': 'Session expired'})
        
        session_data = current_session[session_id]
        detected_faces = session_data['detected_faces']
        
        # Find the selected face and get matches
        selected_face = None
        for face in detected_faces:
            if face['face_id'] == face_id:
                selected_face = face
                break
        
        if selected_face:
            search_result = face_system.get_matches(selected_face['encoding'])
        else:
            search_result = {'success': False, 'error': 'Face not found'}
        return jsonify(search_result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    stats = face_system.get_statistics()
    return jsonify(stats)

@app.route('/settings')
def settings():
    """Settings page"""
    current_tolerance = face_system.config.default_tolerance
    stats = face_system.get_statistics()
    return render_template('settings.html', 
                         current_tolerance=current_tolerance, 
                         stats=stats)

@app.route('/settings', methods=['POST'])
def update_settings():
    """Update settings"""
    try:
        new_tolerance = float(request.form.get('tolerance', 0.6))
        face_system.update_config({'default_tolerance': new_tolerance})
        flash(f'Search tolerance updated to {new_tolerance}')
    except ValueError:
        flash('Invalid tolerance value')
    
    return redirect(url_for('settings'))

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 16MB.')
    return redirect(url_for('upload_file'))

if __name__ == '__main__':
    # Create required directories
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
