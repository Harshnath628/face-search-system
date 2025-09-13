"""
Core Face Search System - High-level API for easy integration.

This module provides a simple, unified interface for all face recognition
and search functionality, making it easy for developers to integrate
face search capabilities into their applications.
"""

import os
import tempfile
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import logging

from .config import Config
from .detector import FaceDetector
from .database import FaceDatabase
from .search_engine import FaceSearchEngine


class FaceSearchSystem:
    """
    High-level API for face recognition and search functionality.
    
    This class provides a simple interface that encapsulates all the complexity
    of face detection, encoding, storage, and search operations.
    
    Example:
        # Basic usage
        face_system = FaceSearchSystem()
        
        # Add a profile
        profile_id = face_system.add_profile(
            name="John Doe",
            image_path="john.jpg",
            metadata={"department": "Engineering"}
        )
        
        # Search for faces in an image
        results = face_system.search_faces_in_image("group_photo.jpg")
        
        # Get matches for a detected face
        matches = face_system.get_matches(results['faces'][0]['encoding'])
    """
    
    def __init__(self, config: Optional[Union[Config, Dict[str, Any], str]] = None):
        """
        Initialize the Face Search System.
        
        Args:
            config: Configuration object, dictionary, or path to config file
        """
        # Handle different config input types
        if isinstance(config, str):
            self.config = Config.from_file(config)
        elif isinstance(config, dict):
            self.config = Config(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            self.config = Config()
        
        # Validate configuration
        is_valid, errors = self.config.validate()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")
        
        # Initialize components
        self.detector = FaceDetector()
        self.database = FaceDatabase(self.config.database_path)
        self.search_engine = FaceSearchEngine(self.config.database_path, self.config.default_tolerance)
        
        # Setup logging if enabled
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging if enabled in config."""
        if self.config.get('log_searches', False):
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger('FaceSearchSystem')
        else:
            self.logger = logging.getLogger('FaceSearchSystem')
            self.logger.setLevel(logging.WARNING)
    
    def add_profile(
        self,
        name: str,
        image_path: str,
        age: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Add a new profile to the face database.
        
        Args:
            name: Person's name
            image_path: Path to the person's image
            age: Person's age (optional)
            description: Description or additional info (optional)
            metadata: Additional metadata dictionary (optional)
        
        Returns:
            Profile ID if successful, None otherwise
            
        Example:
            profile_id = face_system.add_profile(
                name="Jane Smith",
                image_path="jane.jpg",
                age="28",
                description="Software Engineer",
                metadata={"department": "AI Research", "employee_id": "12345"}
            )
        """
        try:
            # Validate image path
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return None
            
            # Get face encoding
            face_encoding = self.detector.get_face_encoding_from_image(image_path)
            if face_encoding is None:
                self.logger.error(f"No face detected in image: {image_path}")
                return None
            
            # Prepare profile data
            profile_data = {
                'name': name,
                'age': age or '',
                'description': description or '',
                'metadata': metadata or {}
            }
            
            # Copy image to database directory (optional, for security)
            if self.config.get('copy_images_to_database', True):
                image_ext = Path(image_path).suffix
                new_image_name = f"profile_{len(self.database.get_all_profiles())}_{name.replace(' ', '_')}{image_ext}"
                new_image_path = os.path.join(self.config.database_path, 'images', new_image_name)
                
                # Ensure images directory exists
                os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
                
                # Copy image
                import shutil
                shutil.copy2(image_path, new_image_path)
                image_path = new_image_path
            
            # Add to database
            profile_id = self.search_engine.add_profile_to_search(profile_data, face_encoding, image_path)
            
            if profile_id:
                self.logger.info(f"Added profile: {name} (ID: {profile_id})")
            else:
                self.logger.error(f"Failed to add profile: {name}")
            
            return profile_id
            
        except Exception as e:
            self.logger.error(f"Error adding profile {name}: {str(e)}")
            return None
    
    def search_faces_in_image(self, image_path: str) -> Dict[str, Any]:
        """
        Detect all faces in an image and return their information.
        
        Args:
            image_path: Path to the image to search
        
        Returns:
            Dictionary containing detected faces and their information
            
        Example:
            results = face_system.search_faces_in_image("group_photo.jpg")
            print(f"Found {results['total_faces']} faces")
            for face in results['faces']:
                print(f"Face {face['face_id']} at coordinates {face['coordinates']}")
        """
        try:
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'error': f'Image file not found: {image_path}',
                    'total_faces': 0,
                    'faces': []
                }
            
            result = self.detector.detect_faces_in_image(image_path)
            
            if result['success']:
                self.logger.info(f"Detected {result['total_faces']} faces in {image_path}")
            else:
                self.logger.error(f"Face detection failed for {image_path}: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error searching faces in {image_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'total_faces': 0,
                'faces': []
            }
    
    def get_matches(
        self,
        face_encoding: Any,
        max_results: Optional[int] = None,
        min_similarity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Find matching profiles for a given face encoding.
        
        Args:
            face_encoding: Face encoding (numpy array)
            max_results: Maximum number of results to return
            min_similarity: Minimum similarity score (0-100)
        
        Returns:
            Dictionary containing matching profiles
            
        Example:
            matches = face_system.get_matches(
                face_encoding=detected_face['encoding'],
                max_results=5,
                min_similarity=70.0
            )
            for match in matches['matches']:
                print(f"{match['profile']['name']}: {match['similarity_score']}% match")
        """
        try:
            max_results = max_results or self.config.max_search_results
            min_similarity = min_similarity or self.config.similarity_threshold
            
            # Perform search
            result = self.search_engine.search_face(face_encoding, max_results)
            
            if result['success']:
                # Filter by minimum similarity
                filtered_matches = [
                    match for match in result['matches']
                    if match['similarity_score'] >= min_similarity
                ]
                result['matches'] = filtered_matches
                result['total_matches'] = len(filtered_matches)
                
                self.logger.info(f"Found {len(filtered_matches)} matches above {min_similarity}% similarity")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting matches: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'matches': [],
                'total_matches': 0
            }
    
    def search_and_match(
        self,
        image_path: str,
        face_id: Optional[int] = None,
        max_results: Optional[int] = None,
        min_similarity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        One-step operation: detect faces in image and find matches.
        
        Args:
            image_path: Path to the image to search
            face_id: Specific face ID to search (if multiple faces detected)
            max_results: Maximum number of results per face
            min_similarity: Minimum similarity score
        
        Returns:
            Dictionary containing detected faces and their matches
            
        Example:
            results = face_system.search_and_match(
                "group_photo.jpg",
                max_results=3,
                min_similarity=75.0
            )
            
            for face_result in results['face_results']:
                print(f"Face {face_result['face_id']}:")
                for match in face_result['matches']:
                    print(f"  - {match['profile']['name']}: {match['similarity_score']}%")
        """
        try:
            # Detect faces
            detection_result = self.search_faces_in_image(image_path)
            
            if not detection_result['success']:
                return detection_result
            
            # If specific face_id requested, only process that face
            faces_to_process = detection_result['faces']
            if face_id is not None:
                faces_to_process = [f for f in faces_to_process if f['face_id'] == face_id]
                if not faces_to_process:
                    return {
                        'success': False,
                        'error': f'Face {face_id} not found in image',
                        'face_results': []
                    }
            
            # Search for matches for each face
            face_results = []
            for face in faces_to_process:
                matches = self.get_matches(
                    face['encoding'],
                    max_results=max_results,
                    min_similarity=min_similarity
                )
                
                face_results.append({
                    'face_id': face['face_id'],
                    'coordinates': face['coordinates'],
                    'cropped_image_path': face['cropped_image_path'],
                    'matches': matches['matches'],
                    'total_matches': matches['total_matches']
                })
            
            return {
                'success': True,
                'original_image': image_path,
                'total_faces_detected': detection_result['total_faces'],
                'faces_processed': len(face_results),
                'face_results': face_results
            }
            
        except Exception as e:
            self.logger.error(f"Error in search_and_match: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'face_results': []
            }
    
    def get_all_profiles(self) -> Dict[str, Any]:
        """
        Get all profiles from the database.
        
        Returns:
            Dictionary of all profiles
        """
        try:
            return self.search_engine.get_all_profiles()
        except Exception as e:
            self.logger.error(f"Error getting all profiles: {str(e)}")
            return {}
    
    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific profile by ID.
        
        Args:
            profile_id: Profile ID to retrieve
        
        Returns:
            Profile data if found, None otherwise
        """
        try:
            return self.search_engine.get_profile(profile_id)
        except Exception as e:
            self.logger.error(f"Error getting profile {profile_id}: {str(e)}")
            return None
    
    def delete_profile(self, profile_id: str) -> bool:
        """
        Delete a profile from the database.
        
        Args:
            profile_id: Profile ID to delete
        
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.search_engine.delete_profile(profile_id)
            if result:
                self.logger.info(f"Deleted profile: {profile_id}")
            else:
                self.logger.error(f"Failed to delete profile: {profile_id}")
            return result
        except Exception as e:
            self.logger.error(f"Error deleting profile {profile_id}: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary containing system statistics
        """
        try:
            stats = self.search_engine.get_search_statistics()
            return stats
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return {}
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration and reinitialize components if needed.
        
        Args:
            updates: Dictionary of configuration updates
        """
        try:
            self.config.update(updates)
            
            # Update search engine tolerance if changed
            if 'default_tolerance' in updates:
                self.search_engine.update_tolerance(updates['default_tolerance'])
            
            self.logger.info(f"Configuration updated: {updates}")
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
    
    def validate_image(self, image_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if an image is suitable for face recognition.
        
        Args:
            image_path: Path to the image to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not os.path.exists(image_path):
                return False, f"Image file not found: {image_path}"
            
            # Check file extension
            file_ext = Path(image_path).suffix.lower()
            if file_ext not in self.config.supported_formats:
                return False, f"Unsupported file format: {file_ext}"
            
            # Check file size
            file_size = os.path.getsize(image_path)
            max_size = self.config.get('max_upload_size', 16 * 1024 * 1024)
            if file_size > max_size:
                return False, f"File too large: {file_size} bytes (max: {max_size})"
            
            # Try to load the image
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    # Check image dimensions
                    width, height = img.size
                    min_size = self.config.min_face_size
                    if width < min_size[0] or height < min_size[1]:
                        return False, f"Image too small: {width}x{height} (min: {min_size[0]}x{min_size[1]})"
            except Exception:
                return False, "Invalid or corrupted image file"
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup if needed."""
        # Perform any cleanup operations
        pass
    
    def __str__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (f"FaceSearchSystem("
                f"profiles={stats.get('total_profiles', 0)}, "
                f"tolerance={self.config.default_tolerance})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"FaceSearchSystem(config={self.config})"
