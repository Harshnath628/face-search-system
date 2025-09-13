# Face Search System

A comprehensive face recognition and search system built with Python, Flask, and OpenCV. Upload images containing faces, detect and identify multiple faces, and search for matching profiles in your database.

## Features

- **Multi-Face Detection**: Automatically detect and identify multiple faces in uploaded images
- **Face Recognition**: Advanced facial recognition using the `face_recognition` library
- **Search Functionality**: Search for faces in your database with adjustable similarity tolerance
- **Profile Management**: Add, view, edit, and delete face profiles
- **Web Interface**: User-friendly web interface with drag-and-drop image uploads
- **Real-time Results**: Instant search results with similarity scores
- **Database Management**: Secure local storage of face encodings and profile data

## How It Works

1. **Upload Image**: Upload an image containing one or more faces
2. **Face Detection**: System automatically detects all faces and labels them as "Face 1", "Face 2", etc.
3. **Select Face**: Choose which detected face you want to search for
4. **Search Database**: System compares the selected face against all stored profiles
5. **View Results**: See matching profiles with similarity scores and detailed information

## Installation

### Prerequisites

- Python 3.7 or higher
- Windows/Linux/macOS
- At least 2GB RAM (recommended)
- 1GB free disk space

### Quick Setup

1. **Clone or Download** this repository to your local machine

2. **Navigate** to the project directory:
   ```bash
   cd face_search_system
   ```

3. **Run the setup script**:
   ```bash
   python setup.py
   ```

   The setup script will:
   - Check Python version compatibility
   - Install all required dependencies
   - Create necessary directories
   - Initialize the database

### Manual Installation

If the setup script fails, install dependencies manually:

```bash
pip install flask==2.3.3
pip install opencv-python==4.8.1.78
pip install face-recognition==1.3.0
pip install Pillow==10.0.1
pip install numpy==1.24.3
pip install werkzeug==2.3.7
```

**Note**: On Windows, you might need to install `dlib` separately:
```bash
pip install cmake
pip install dlib
```

## Usage

### Starting the Application

1. Navigate to the project directory
2. Run the Flask application:
   ```bash
   python app.py
   ```
3. Open your web browser and go to: `http://localhost:5000`

### Basic Workflow

#### 1. Add Profiles to Database
- Click "Add New Profile" in the navigation menu
- Fill in the person's information (name, age, description)
- Upload a clear photo of their face
- Submit the form to add them to the database

#### 2. Search for Faces
- Click "Search by Image" in the navigation menu
- Upload an image containing faces you want to search for
- The system will detect all faces and show them with labels
- Click "Search This Face" on the face you want to find matches for
- View the search results with similarity scores

#### 3. Manage Profiles
- Click "View Profiles" to see all stored profiles
- Search, filter, and sort profiles
- Click on any profile to view detailed information
- Delete profiles as needed

### Advanced Features

#### Adjusting Search Tolerance
- Go to Settings to adjust the face recognition tolerance
- Lower values (0.1-0.3) = More strict matching
- Higher values (0.7-1.0) = More lenient matching
- Balanced setting (0.4-0.6) works well for most cases

#### API Endpoints
The system also provides REST API endpoints:
- `POST /api/search_face` - Search for faces programmatically
- `GET /api/stats` - Get system statistics

## File Structure

```
face_search_system/
├── app.py                 # Main Flask application
├── face_detector.py       # Face detection module
├── face_database.py       # Database management
├── face_search.py         # Search engine
├── setup.py              # Setup script
├── requirements.txt       # Dependencies
├── README.md             # This file
├── templates/            # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── face_selection.html
│   ├── search_results.html
│   ├── add_profile.html
│   ├── profiles.html
│   ├── profile_detail.html
│   └── settings.html
├── uploads/              # Uploaded images
├── database/             # Face database files
│   ├── profiles.json     # Profile metadata
│   └── face_encodings.pkl # Face encodings
└── static/               # Static files (CSS, JS)
```

## Theory and Implementation Details

This section provides a comprehensive understanding of the theoretical concepts and algorithms used in building this face recognition system. By the end of this section, you'll understand every component from basic image processing to advanced neural networks.

### 1. Computer Vision Fundamentals

#### What is Computer Vision?
Computer Vision is a field of artificial intelligence that enables computers to interpret and understand visual information from the world. It involves:
- **Image Processing**: Manipulating and analyzing digital images
- **Pattern Recognition**: Identifying patterns and structures in visual data
- **Feature Extraction**: Finding important characteristics in images
- **Object Detection**: Locating and identifying objects in images

#### Digital Image Representation
```python
# Images are represented as multi-dimensional arrays
# Grayscale: 2D array (height × width)
# Color (RGB): 3D array (height × width × channels)
import numpy as np
image_gray = np.array([[120, 130, 140], [110, 115, 125]])  # 2×3 grayscale
image_rgb = np.array([[[255,0,0], [0,255,0]], [[0,0,255], [255,255,0]]])  # 2×2×3 RGB
```

#### Color Spaces
- **RGB**: Red, Green, Blue - additive color model
- **BGR**: Blue, Green, Red - OpenCV's default format
- **HSV**: Hue, Saturation, Value - better for color-based segmentation
- **Grayscale**: Single channel, reduces computational complexity

### 2. Image Processing Techniques

#### Filtering and Enhancement
```python
import cv2

# Gaussian Blur - reduces noise and smooths images
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Histogram Equalization - improves contrast
enhanced = cv2.equalizeHist(gray_image)

# Edge Detection - finds boundaries
edges = cv2.Canny(image, 50, 150)
```

#### Mathematical Operations
- **Convolution**: Applies filters to extract features
- **Morphological Operations**: Shape analysis and noise removal
- **Geometric Transformations**: Rotation, scaling, translation

### 3. Face Detection Theory

#### Haar Cascades (Traditional Method)
Haar cascades use machine learning to detect objects:
```python
# Haar features are simple rectangular features
# Examples: edge features, line features, four-rectangle features
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
```

**How Haar Cascades Work:**
1. **Feature Selection**: Use simple rectangular features
2. **Integral Image**: Fast computation of rectangular sums
3. **AdaBoost Learning**: Combines weak classifiers into strong ones
4. **Cascade Structure**: Series of classifiers, reject non-faces early

#### Modern Deep Learning Approaches
Our system uses more advanced methods:

**HOG (Histogram of Oriented Gradients) + Linear SVM:**
```python
# HOG features capture edge and gradient structure
# Used in dlib's face detector
detector = dlib.get_frontal_face_detector()
faces = detector(gray_image)
```

### 4. Face Recognition Theory

#### Traditional Approaches

**1. Eigenfaces (PCA-based)**
```python
# Principal Component Analysis for dimensionality reduction
# Represents faces as linear combinations of "eigenfaces"
import numpy as np
from sklearn.decomposition import PCA

# Example concept (simplified)
face_matrix = np.array([face1_vector, face2_vector, ...])  # Each row is a flattened face
pca = PCA(n_components=50)  # Reduce to 50 principal components
eigenfaces = pca.fit_transform(face_matrix)
```

**2. Fisherfaces (LDA-based)**
```python
# Linear Discriminant Analysis - maximizes class separability
# Better than PCA for classification tasks
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
fisherfaces = lda.fit_transform(face_matrix, labels)
```

**3. Local Binary Patterns (LBP)**
```python
# Describes local texture patterns
# Robust to illumination changes
def lbp_histogram(image):
    # Convert each pixel to LBP code based on neighbors
    # Create histogram of LBP codes
    pass
```

#### Deep Learning Revolution

**Convolutional Neural Networks (CNNs)**
Modern face recognition uses deep CNNs that learn hierarchical features:

```python
# Conceptual CNN architecture for face recognition
class FaceRecognitionCNN:
    def __init__(self):
        self.conv_layers = [
            Conv2D(32, (3,3), activation='relu'),    # Low-level features (edges)
            Conv2D(64, (3,3), activation='relu'),    # Mid-level features (textures)
            Conv2D(128, (3,3), activation='relu'),   # High-level features (face parts)
            Conv2D(256, (3,3), activation='relu')    # Complex features (face identity)
        ]
        self.fc_layers = [
            Dense(512, activation='relu'),           # Feature combination
            Dense(128, activation='linear')          # Face embedding
        ]
```

**What Each Layer Learns:**
- **Layer 1**: Edges and simple patterns
- **Layer 2**: Textures and simple shapes
- **Layer 3**: Face parts (eyes, nose, mouth)
- **Layer 4**: Complete facial features
- **Final Layer**: 128-dimensional face embedding

### 5. Face Encoding and Embeddings

#### What is a Face Encoding?
A face encoding is a numerical representation (vector) of a face that captures its unique characteristics:

```python
# Face encoding is typically a 128-dimensional vector
face_encoding = [0.1, -0.3, 0.7, 0.2, ...]  # 128 numbers

# These numbers represent learned facial features:
# - Eye shape and position
# - Nose characteristics
# - Mouth features
# - Facial geometry
# - Skin texture patterns
```

#### How Face Encodings are Generated

**1. Face Detection**: Locate face in image
```python
# Our face_detector.py implementation
face_locations = face_recognition.face_locations(image)
# Returns: [(top, right, bottom, left), ...]
```

**2. Face Alignment**: Normalize face orientation
```python
# Align face to standard position
# - Center the face
# - Rotate to upright position
# - Scale to consistent size
aligned_face = align_face(cropped_face)
```

**3. Feature Extraction**: Pass through neural network
```python
# Deep neural network extracts 128-dimensional encoding
face_encoding = face_recognition.face_encodings(image, face_locations)[0]
print(f"Encoding shape: {face_encoding.shape}")  # (128,)
```

#### Mathematical Properties of Face Encodings

**Euclidean Distance**: Measures similarity between faces
```python
import numpy as np

def face_distance(encoding1, encoding2):
    """Calculate Euclidean distance between face encodings"""
    return np.linalg.norm(encoding1 - encoding2)

# Distance interpretation:
# 0.0 = Identical faces
# 0.6 = Default threshold (same person)
# 1.0+ = Different people
distance = face_distance(face1_encoding, face2_encoding)
similarity_percentage = max(0, (1 - distance) * 100)
```

**Cosine Similarity**: Alternative similarity measure
```python
def cosine_similarity(encoding1, encoding2):
    """Calculate cosine similarity between face encodings"""
    dot_product = np.dot(encoding1, encoding2)
    norm1 = np.linalg.norm(encoding1)
    norm2 = np.linalg.norm(encoding2)
    return dot_product / (norm1 * norm2)
```

### 6. Deep Learning Architecture Details

#### The face_recognition Library Architecture

Our system uses the `face_recognition` library, which implements:

**1. Face Detection: HOG + Linear SVM**
```python
# Based on dlib's implementation
# HOG (Histogram of Oriented Gradients) features
# Linear SVM classifier
def detect_faces_hog(image):
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Calculate HOG features
    hog_features = calculate_hog_features(gray)
    
    # 3. Apply sliding window
    # 4. Classify each window with SVM
    # 5. Non-maximum suppression
    return face_locations
```

**2. Face Recognition: ResNet-based CNN**
```python
# Based on ResNet architecture with 29 layers
# Trained on millions of face images
class ResNetFaceRecognition:
    def __init__(self):
        # Residual blocks allow very deep networks
        self.residual_blocks = [
            ResidualBlock(64, 64),   # Early feature extraction
            ResidualBlock(64, 128),  # Increasing complexity
            ResidualBlock(128, 256), # High-level features
            ResidualBlock(256, 512)  # Face-specific features
        ]
        self.output_layer = Dense(128)  # 128-dim embedding
    
    def forward(self, face_image):
        # Input: 150x150x3 face image
        # Output: 128-dimensional face encoding
        pass
```

#### Training Process (Conceptual)

**1. Dataset Preparation**
```python
# Large dataset with multiple images per person
dataset = {
    'person_1': [image1, image2, image3, ...],
    'person_2': [image1, image2, image3, ...],
    # ... millions of people
}
```

**2. Triplet Loss Function**
```python
def triplet_loss(anchor, positive, negative, margin=0.2):
    """Triplet loss for face recognition training"""
    # Anchor: reference face
    # Positive: same person, different image
    # Negative: different person
    
    pos_distance = np.linalg.norm(anchor - positive)
    neg_distance = np.linalg.norm(anchor - negative)
    
    loss = max(0, pos_distance - neg_distance + margin)
    return loss

# Goal: minimize distance between same person's faces
#       maximize distance between different people's faces
```

### 7. Database and Storage Theory

#### Why Not Store Images Directly?

**Problems with Image Storage:**
- Large file sizes (MB per image)
- Slow comparison (pixel-by-pixel)
- Sensitive to lighting, angle changes
- Privacy concerns

**Benefits of Encoding Storage:**
```python
# Face encoding: 128 floats × 4 bytes = 512 bytes
# Original image: 150×150×3 bytes = 67,500 bytes
# Compression ratio: 131:1

face_encoding = np.array([0.1, -0.2, 0.5, ...])  # 128 dimensions
# vs
face_image = np.array([[[255,0,0], [0,255,0]], ...])  # 150×150×3
```

#### Database Schema Design

**profiles.json Structure:**
```json
{
  "profile_20241213_143052_123456": {
    "id": "profile_20241213_143052_123456",
    "name": "John Doe",
    "age": "30",
    "description": "Software Engineer",
    "image_path": "uploads/uuid_profile_john.jpg",
    "created_date": "2024-12-13T14:30:52.123456",
    "metadata": {
      "uploaded_by": "web_interface",
      "original_filename": "john_photo.jpg"
    }
  }
}
```

**face_encodings.pkl Structure:**
```python
# Python pickle format for efficient numpy array storage
encodings_dict = {
    "profile_20241213_143052_123456": numpy.array([0.1, -0.2, ...]),  # 128-dim
    "profile_20241213_143053_789012": numpy.array([0.3, 0.1, ...]),   # 128-dim
    # ...
}
```

### 8. Search Algorithm Theory

#### Nearest Neighbor Search

Our search algorithm implements k-nearest neighbors in 128-dimensional space:

```python
def search_face_detailed(query_encoding, stored_encodings, k=5):
    """Detailed search implementation"""
    distances = []
    
    # Calculate distance to each stored encoding
    for profile_id, stored_encoding in stored_encodings.items():
        # Euclidean distance in 128-dimensional space
        distance = np.sqrt(np.sum((query_encoding - stored_encoding) ** 2))
        
        # Convert distance to similarity percentage
        similarity = max(0, (1 - distance) * 100)
        
        distances.append({
            'profile_id': profile_id,
            'distance': distance,
            'similarity': similarity
        })
    
    # Sort by similarity (ascending distance)
    distances.sort(key=lambda x: x['distance'])
    
    # Apply threshold filtering
    threshold = 0.6  # Configurable tolerance
    matches = [d for d in distances if d['distance'] <= threshold]
    
    return matches[:k]  # Return top k matches
```

#### Optimization Techniques

**1. Early Termination**
```python
# Stop search if we find very close match
if distance < 0.3:  # Very high confidence
    return [current_match]  # Don't search further
```

**2. Indexing (for large databases)**
```python
# For millions of faces, use approximate nearest neighbors
from sklearn.neighbors import NearestNeighbors

class OptimizedFaceSearch:
    def __init__(self, encodings):
        self.nn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.nn_model.fit(encodings)
    
    def search(self, query_encoding):
        distances, indices = self.nn_model.kneighbors([query_encoding])
        return distances, indices
```

### 9. Web Application Architecture

#### MVC Pattern Implementation

**Model Layer** (`face_database.py`, `face_search.py`):
```python
# Data management and business logic
class FaceDatabase:  # Model for data persistence
    def add_profile(self, profile_data, face_encoding, image_path):
        # Data validation and storage logic
        pass

class FaceSearchEngine:  # Model for search operations
    def search_face(self, query_encoding, max_results=5):
        # Search algorithm implementation
        pass
```

**View Layer** (HTML Templates):
```html
<!-- Presentation layer - what user sees -->
<div class="face-box">
    <img src="{{ face.cropped_image_path }}" />
    <button onclick="searchFace({{ face.face_id }})">Search This Face</button>
</div>
```

**Controller Layer** (`app.py`):
```python
# Request handling and coordination
@app.route('/search/<session_id>/<int:face_id>')
def search_face(session_id, face_id):
    # 1. Validate input
    # 2. Call model methods
    # 3. Prepare data for view
    # 4. Return response
    search_result = search_engine.search_by_face_id(detected_faces, face_id)
    return render_template('search_results.html', search_result=search_result)
```

#### Session Management

```python
# Temporary storage for multi-step process
current_session = {
    'session_uuid': {
        'original_image': '/path/to/uploaded/image.jpg',
        'detected_faces': [
            {
                'face_id': 1,
                'location': (top, right, bottom, left),
                'encoding': numpy.array([...]),  # 128-dim
                'cropped_image_path': '/path/to/face1.jpg'
            },
            # ... more faces
        ],
        'total_faces': 3
    }
}
```

### 10. Performance Considerations

#### Computational Complexity

**Face Detection**: O(n×m) where n×m is image size
- HOG feature calculation: Linear in image pixels
- Sliding window: Depends on window stride
- Non-maximum suppression: O(k²) where k is number of detections

**Face Encoding**: O(1) per face
- Fixed neural network computation
- Independent of database size

**Face Search**: O(n) where n is number of stored profiles
- Linear search through all encodings
- Each comparison: O(128) for 128-dimensional vectors

#### Memory Usage

```python
# Memory requirements calculation
def calculate_memory_usage(num_profiles):
    # Per profile storage:
    encoding_size = 128 * 4  # 128 floats × 4 bytes = 512 bytes
    metadata_size = 500      # Estimated JSON overhead
    image_size = 50 * 1024   # Average 50KB per profile image
    
    total_per_profile = encoding_size + metadata_size + image_size
    total_memory = num_profiles * total_per_profile
    
    return {
        'per_profile_bytes': total_per_profile,
        'total_mb': total_memory / (1024 * 1024),
        'estimated_profiles_per_gb': (1024 * 1024 * 1024) / total_per_profile
    }

# Example: 10,000 profiles ≈ 500MB total storage
```

### 11. Error Handling and Edge Cases

#### Common Challenges and Solutions

**1. No Face Detected**
```python
if len(face_locations) == 0:
    return {
        'success': False,
        'error': 'No faces detected in image',
        'suggestions': [
            'Ensure face is clearly visible',
            'Check image quality and lighting',
            'Try a different angle'
        ]
    }
```

**2. Multiple Faces - Disambiguation**
```python
# Let user choose which face to search
if len(face_locations) > 1:
    return render_template('face_selection.html', 
                         detected_faces=detected_faces)
```

**3. Poor Quality Encodings**
```python
def validate_encoding_quality(face_encoding, face_image):
    """Check if encoding is reliable"""
    # Check encoding magnitude
    encoding_norm = np.linalg.norm(face_encoding)
    if encoding_norm < 0.1 or encoding_norm > 10:
        return False, "Encoding magnitude unusual"
    
    # Check face image quality
    if face_image.shape[0] < 50 or face_image.shape[1] < 50:
        return False, "Face image too small"
    
    return True, "Encoding quality acceptable"
```

### 12. Security and Privacy Considerations

#### Data Protection

**1. Local Storage Only**
```python
# All data stored locally - no external transmission
# Face encodings are mathematical representations, not images
# Original images can be deleted after encoding extraction
```

**2. Encoding Irreversibility**
```python
# Face encodings cannot be reverse-engineered to reconstruct original image
# 128 numbers cannot recreate 67,500 pixel values (150×150×3)
# Provides privacy protection while enabling recognition
```

**3. Access Control**
```python
# Implement user authentication for production use
@app.before_request
def require_authentication():
    # Check user credentials
    # Implement session management
    # Log access attempts
    pass
```

### 13. Advanced Topics and Extensions

#### Real-time Processing

```python
# Video stream processing
import cv2

def process_video_stream():
    cap = cv2.VideoCapture(0)  # Webcam
    
    while True:
        ret, frame = cap.read()
        
        # Detect faces in current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        # Search for matches
        for encoding in face_encodings:
            matches = search_engine.search_face(encoding)
            # Display results on frame
        
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

#### Scalability Improvements

**1. Approximate Nearest Neighbors**
```python
# For large databases (millions of faces)
from annoy import AnnoyIndex

class ScalableFaceSearch:
    def __init__(self, dimension=128):
        self.index = AnnoyIndex(dimension, 'euclidean')
        self.profile_map = {}
    
    def add_profile(self, profile_id, encoding):
        item_id = len(self.profile_map)
        self.index.add_item(item_id, encoding)
        self.profile_map[item_id] = profile_id
    
    def build_index(self):
        self.index.build(10)  # 10 trees
    
    def search(self, query_encoding, k=5):
        similar_items = self.index.get_nns_by_vector(query_encoding, k)
        return [self.profile_map[item] for item in similar_items]
```

**2. Distributed Processing**
```python
# Horizontal scaling with multiple servers
from celery import Celery

app = Celery('face_search')

@app.task
def process_face_encoding(image_data):
    """Background task for face processing"""
    # Detect faces
    # Extract encodings
    # Store in distributed database
    return results
```

#### Machine Learning Pipeline

```python
# Custom model training pipeline
class FaceRecognitionPipeline:
    def __init__(self):
        self.data_loader = None
        self.model = None
        self.loss_function = None
        self.optimizer = None
    
    def prepare_data(self, dataset_path):
        # Load and preprocess training data
        # Data augmentation
        # Create triplets for triplet loss
        pass
    
    def train_model(self, epochs=100):
        # Training loop
        # Validation
        # Model checkpointing
        pass
    
    def evaluate_model(self, test_dataset):
        # Performance metrics
        # Accuracy, precision, recall
        # ROC curves
        pass
```

This comprehensive theory section covers everything from basic computer vision concepts to advanced machine learning techniques used in the face search system. Each concept is explained with code examples and mathematical foundations, providing a complete understanding of the system's inner workings.

## Technical Details

### Face Recognition Technology
- Uses the `face_recognition` library built on dlib
- Employs deep neural networks for face encoding
- Achieves high accuracy with proper image quality
- Supports multiple faces per image

### Database Storage
- Profile metadata stored in JSON format
- Face encodings stored using Python pickle
- Local file-based storage (no external database required)
- Automatic backup and recovery

### Security Considerations
- All data stored locally on your system
- No data transmitted to external servers
- Face encodings are mathematical representations (not actual images)
- User-controlled access and data management

## Tips for Best Results

### Image Quality
- Use clear, well-lit images
- Ensure faces are clearly visible
- Avoid heavy shadows or backlighting
- Higher resolution images work better

### Face Position
- Front-facing or near-front angles work best
- Avoid extreme side angles
- Ensure eyes, nose, and mouth are visible
- Remove sunglasses, masks, or obstructions

### Database Management
- Add multiple photos of the same person for better recognition
- Use descriptive names and information
- Regularly review and clean up the database
- Adjust tolerance settings based on your needs

## Troubleshooting

### Common Issues

**1. Installation Problems**
- Ensure Python 3.7+ is installed
- Try installing dependencies one by one
- On Windows, install Visual Studio Build Tools if needed

**2. Face Detection Not Working**
- Check image quality and lighting
- Ensure faces are clearly visible
- Try different image formats (JPG, PNG)
- Verify the image file isn't corrupted

**3. Low Accuracy Results**
- Adjust the tolerance settings
- Add more photos of the same person
- Improve image quality
- Check for proper face alignment

**4. Performance Issues**
- Close other applications to free memory
- Use smaller image files
- Reduce the number of profiles if database is very large

### Getting Help

If you encounter issues:
1. Check the browser console for error messages
2. Look at the terminal/command prompt for Python errors
3. Verify all dependencies are properly installed
4. Ensure sufficient disk space and memory

## System Requirements

### Minimum Requirements
- Python 3.7+
- 4GB RAM
- 1GB free disk space
- Modern web browser

### Recommended Requirements
- Python 3.8+
- 8GB RAM
- 2GB free disk space
- Chrome/Firefox/Edge browser

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Submitting pull requests
- Improving documentation

## Changelog

### Version 1.0.0
- Initial release
- Multi-face detection and recognition
- Web-based interface
- Profile management system
- Adjustable search tolerance
- Local database storage

---

**Note**: This system is designed for educational and personal use. Ensure compliance with privacy laws and regulations in your jurisdiction when using facial recognition technology.
