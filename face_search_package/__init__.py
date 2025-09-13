"""
Face Search Package

A comprehensive face recognition and search system that can be easily integrated
into any Python project. Provides face detection, encoding, and similarity search
capabilities with a simple, clean API.

Example usage:
    from face_search_package import FaceSearchSystem
    
    # Initialize the system
    face_system = FaceSearchSystem()
    
    # Add a profile
    profile_id = face_system.add_profile(
        name="John Doe",
        image_path="path/to/john.jpg",
        metadata={"age": 30, "department": "Engineering"}
    )
    
    # Search for faces in an image
    results = face_system.search_faces_in_image("path/to/group_photo.jpg")
    
    # Get matches for a specific detected face
    matches = face_system.get_matches(results['faces'][0]['encoding'])
"""

__version__ = "1.0.0"
__author__ = "Harsh Nath Tripathi"
__email__ = "harshnath628@gmail.com"

# Import main classes for easy access
from .core import FaceSearchSystem
from .detector import FaceDetector
from .database import FaceDatabase
from .search_engine import FaceSearchEngine
from .config import Config

# Public API
__all__ = [
    'FaceSearchSystem',
    'FaceDetector', 
    'FaceDatabase',
    'FaceSearchEngine',
    'Config'
]
