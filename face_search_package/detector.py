import face_recognition
import cv2
import numpy as np
from PIL import Image
import os
from typing import List, Dict, Any, Optional, Tuple
from .config import Config

class FaceDetector:
    def __init__(self):
        """Initialize the face detector"""
        pass
    
    def detect_faces_in_image(self, image_path):
        """
        Detect all faces in an image and return face locations and encodings
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            dict: Dictionary containing face locations, encodings, and cropped face images
        """
        try:
            # Load the image
            image = face_recognition.load_image_file(image_path)
            
            # Find all face locations and encodings
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            # Convert to PIL Image for easier manipulation
            pil_image = Image.fromarray(image)
            
            detected_faces = []
            
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                # Extract face coordinates (top, right, bottom, left)
                top, right, bottom, left = face_location
                
                # Crop the face from the image
                face_image = pil_image.crop((left, top, right, bottom))
                
                # Save the cropped face
                face_filename = f"detected_face_{i+1}.jpg"
                face_path = os.path.join("uploads", face_filename)
                face_image.save(face_path)
                
                face_data = {
                    'face_id': i + 1,
                    'location': face_location,
                    'encoding': face_encoding,
                    'cropped_image_path': face_path,
                    'coordinates': {
                        'top': top,
                        'right': right,
                        'bottom': bottom,
                        'left': left
                    }
                }
                
                detected_faces.append(face_data)
            
            return {
                'success': True,
                'total_faces': len(detected_faces),
                'faces': detected_faces,
                'original_image_path': image_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'total_faces': 0,
                'faces': []
            }
    
    def draw_face_boxes(self, image_path, face_locations):
        """
        Draw bounding boxes around detected faces
        
        Args:
            image_path (str): Path to the original image
            face_locations (list): List of face location tuples
            
        Returns:
            str: Path to the image with face boxes drawn
        """
        try:
            # Load image with OpenCV
            image = cv2.imread(image_path)
            
            # Draw rectangles around faces
            for i, (top, right, bottom, left) in enumerate(face_locations):
                # Draw rectangle
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Add face label
                cv2.putText(image, f'Face {i+1}', (left, top-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Save the annotated image
            annotated_path = image_path.replace('.', '_annotated.')
            cv2.imwrite(annotated_path, image)
            
            return annotated_path
            
        except Exception as e:
            print(f"Error drawing face boxes: {str(e)}")
            return image_path
    
    def get_face_encoding_from_image(self, image_path):
        """
        Get face encoding from a single face image
        
        Args:
            image_path (str): Path to face image
            
        Returns:
            numpy.ndarray or None: Face encoding if successful, None otherwise
        """
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                return face_encodings[0]
            else:
                return None
                
        except Exception as e:
            print(f"Error getting face encoding: {str(e)}")
            return None
