"""
Integration Guide - Face Search Package

This guide shows how to integrate the face search package into different
types of applications and projects.

Examples include:
1. Web applications (Flask/Django)
2. Desktop applications
3. Batch processing scripts
4. API services
5. Real-time processing
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_search_package import FaceSearchSystem, Config, ConfigPresets


class WebAppIntegration:
    """Example integration for a web application."""
    
    def __init__(self):
        # Use production configuration for web apps
        config = ConfigPresets.production()
        config.update({
            'database_path': '/app/data/faces',
            'log_searches': True,
            'secure_deletion': True
        })
        self.face_system = FaceSearchSystem(config)
    
    def handle_profile_upload(self, name, image_file, metadata=None):
        """Handle profile upload in a web application."""
        try:
            # Save uploaded file temporarily
            temp_path = f"/tmp/upload_{name.replace(' ', '_')}.jpg"
            image_file.save(temp_path)
            
            # Validate the image
            is_valid, error = self.face_system.validate_image(temp_path)
            if not is_valid:
                return {'success': False, 'error': error}
            
            # Add profile to database
            profile_id = self.face_system.add_profile(
                name=name,
                image_path=temp_path,
                metadata=metadata or {}
            )
            
            # Clean up temporary file
            os.remove(temp_path)
            
            if profile_id:
                return {'success': True, 'profile_id': profile_id}
            else:
                return {'success': False, 'error': 'Failed to add profile'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def handle_search_request(self, image_file, min_similarity=70):
        """Handle search request from web interface."""
        try:
            # Save uploaded image temporarily
            temp_path = f"/tmp/search_{os.urandom(8).hex()}.jpg"
            image_file.save(temp_path)
            
            # Perform search
            results = self.face_system.search_and_match(
                temp_path,
                min_similarity=min_similarity
            )
            
            # Clean up
            os.remove(temp_path)
            
            return results
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


class BatchProcessor:
    """Example batch processing integration."""
    
    def __init__(self, config_file=None):
        # Load configuration from file if provided
        if config_file and os.path.exists(config_file):
            self.face_system = FaceSearchSystem(config_file)
        else:
            # Use memory-efficient configuration for batch processing
            config = ConfigPresets.memory_efficient()
            self.face_system = FaceSearchSystem(config)
    
    def process_directory(self, directory_path, output_file="results.json"):
        """Process all images in a directory and save results."""
        results = []
        image_files = []
        
        # Find all image files
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            image_files.extend(Path(directory_path).glob(f"*{ext}"))
        
        print(f"Processing {len(image_files)} images...")
        
        for i, image_path in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {image_path.name}")
            
            try:
                # Search for faces in the image
                result = self.face_system.search_and_match(
                    str(image_path),
                    min_similarity=60.0
                )
                
                # Store results
                results.append({
                    'image_path': str(image_path),
                    'filename': image_path.name,
                    'result': result
                })
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'filename': image_path.name,
                    'error': str(e)
                })
        
        # Save results to JSON file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {output_file}")
        return results
    
    def bulk_add_profiles(self, profiles_data):
        """Add multiple profiles from a data structure."""
        added_profiles = []
        
        for profile_data in profiles_data:
            try:
                profile_id = self.face_system.add_profile(
                    name=profile_data['name'],
                    image_path=profile_data['image_path'],
                    age=profile_data.get('age'),
                    description=profile_data.get('description'),
                    metadata=profile_data.get('metadata', {})
                )
                
                if profile_id:
                    added_profiles.append({
                        'name': profile_data['name'],
                        'profile_id': profile_id,
                        'status': 'success'
                    })
                else:
                    added_profiles.append({
                        'name': profile_data['name'],
                        'status': 'failed',
                        'error': 'Could not add profile'
                    })
                    
            except Exception as e:
                added_profiles.append({
                    'name': profile_data['name'],
                    'status': 'error',
                    'error': str(e)
                })
        
        return added_profiles


class APIService:
    """Example REST API service integration."""
    
    def __init__(self):
        # Use fast processing for API responses
        config = ConfigPresets.fast_processing()
        config.update({
            'log_searches': True,
            'max_search_results': 20
        })
        self.face_system = FaceSearchSystem(config)
    
    def add_profile_endpoint(self, request_data):
        """API endpoint to add a new profile."""
        try:
            # Validate required fields
            required_fields = ['name', 'image_data']
            for field in required_fields:
                if field not in request_data:
                    return {
                        'success': False,
                        'error': f'Missing required field: {field}'
                    }
            
            # Decode image data (assuming base64 encoded)
            import base64
            import tempfile
            
            image_data = base64.b64decode(request_data['image_data'])
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(image_data)
                temp_path = temp_file.name
            
            try:
                # Add profile
                profile_id = self.face_system.add_profile(
                    name=request_data['name'],
                    image_path=temp_path,
                    age=request_data.get('age'),
                    description=request_data.get('description'),
                    metadata=request_data.get('metadata', {})
                )
                
                return {
                    'success': True,
                    'profile_id': profile_id,
                    'message': 'Profile added successfully'
                }
                
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def search_endpoint(self, request_data):
        """API endpoint to search for faces."""
        try:
            # Validate request
            if 'image_data' not in request_data:
                return {
                    'success': False,
                    'error': 'Missing image_data field'
                }
            
            # Decode image
            import base64
            import tempfile
            
            image_data = base64.b64decode(request_data['image_data'])
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(image_data)
                temp_path = temp_file.name
            
            try:
                # Perform search
                results = self.face_system.search_and_match(
                    temp_path,
                    max_results=request_data.get('max_results', 10),
                    min_similarity=request_data.get('min_similarity', 60.0)
                )
                
                return results
                
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


class DesktopAppIntegration:
    """Example desktop application integration."""
    
    def __init__(self, app_data_dir=None):
        # Use application data directory
        if app_data_dir is None:
            app_data_dir = os.path.expanduser("~/.face_search_app")
        
        config = {
            'database_path': os.path.join(app_data_dir, 'faces'),
            'log_searches': True,
            'default_tolerance': 0.6
        }
        
        self.face_system = FaceSearchSystem(config)
        self.app_data_dir = app_data_dir
    
    def import_photos_from_folder(self, folder_path, progress_callback=None):
        """Import photos from a folder with progress tracking."""
        image_files = []
        
        # Find all image files
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            image_files.extend(Path(folder_path).glob(f"**/*{ext}"))
        
        imported_count = 0
        total_files = len(image_files)
        
        for i, image_path in enumerate(image_files):
            # Update progress
            if progress_callback:
                progress_callback(i, total_files, f"Processing {image_path.name}")
            
            try:
                # Try to add as profile (assuming filename contains name)
                name = image_path.stem.replace('_', ' ').title()
                
                profile_id = self.face_system.add_profile(
                    name=name,
                    image_path=str(image_path),
                    metadata={'imported_from': str(folder_path)}
                )
                
                if profile_id:
                    imported_count += 1
                    
            except Exception as e:
                print(f"Could not import {image_path}: {e}")
        
        if progress_callback:
            progress_callback(total_files, total_files, f"Imported {imported_count} profiles")
        
        return imported_count
    
    def export_database(self, export_path):
        """Export database to a backup file."""
        try:
            import shutil
            import zipfile
            
            # Create zip file with database contents
            with zipfile.ZipFile(export_path, 'w') as zip_file:
                database_path = Path(self.face_system.config.database_path)
                
                for file_path in database_path.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(database_path)
                        zip_file.write(file_path, arcname)
            
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            return False


def demonstrate_integrations():
    """Demonstrate various integration patterns."""
    print("=" * 60)
    print("Face Search Package - Integration Examples")
    print("=" * 60)
    
    # 1. Web Application Integration
    print("\\n1. Web Application Integration:")
    print("   - Production-ready configuration")
    print("   - File upload handling")
    print("   - Temporary file management")
    print("   - Error handling and validation")
    
    # 2. Batch Processing Integration
    print("\\n2. Batch Processing Integration:")
    print("   - Memory-efficient configuration")
    print("   - Directory processing")
    print("   - Progress tracking")
    print("   - Results export")
    
    # 3. API Service Integration
    print("\\n3. API Service Integration:")
    print("   - Fast processing configuration")
    print("   - Base64 image handling")
    print("   - RESTful endpoints")
    print("   - JSON responses")
    
    # 4. Desktop Application Integration
    print("\\n4. Desktop Application Integration:")
    print("   - User data directory")
    print("   - Photo import wizard")
    print("   - Progress callbacks")
    print("   - Database backup/restore")
    
    print("\\n" + "=" * 60)
    print("Integration patterns demonstrated!")
    print("Choose the pattern that best fits your use case.")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_integrations()
