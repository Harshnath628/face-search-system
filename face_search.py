import face_recognition
import numpy as np
from face_database import FaceDatabase

class FaceSearchEngine:
    def __init__(self, database_path="database", tolerance=0.6):
        """
        Initialize the face search engine
        
        Args:
            database_path (str): Path to the face database
            tolerance (float): Face matching tolerance (lower = more strict)
        """
        self.database = FaceDatabase(database_path)
        self.tolerance = tolerance
    
    def search_face(self, query_face_encoding, max_results=5):
        """
        Search for matching faces in the database
        
        Args:
            query_face_encoding (numpy.ndarray): Face encoding to search for
            max_results (int): Maximum number of results to return
            
        Returns:
            list: List of matching profiles with similarity scores
        """
        try:
            # Get all stored face encodings
            stored_encodings = self.database.get_all_face_encodings()
            
            if not stored_encodings:
                return {
                    'success': True,
                    'matches': [],
                    'message': 'No profiles in database to search against'
                }
            
            matches = []
            
            # Compare query face with each stored face
            for profile_id, stored_encoding in stored_encodings.items():
                # Calculate face distance (lower = more similar)
                face_distance = face_recognition.face_distance([stored_encoding], query_face_encoding)[0]
                
                # Convert distance to similarity percentage
                similarity = max(0, (1 - face_distance) * 100)
                
                # Check if it's a match based on tolerance
                is_match = face_distance <= self.tolerance
                
                if is_match:
                    # Get profile information
                    profile = self.database.get_profile(profile_id)
                    
                    if profile:
                        match_data = {
                            'profile_id': profile_id,
                            'profile': profile,
                            'similarity_score': round(similarity, 2),
                            'face_distance': round(face_distance, 4),
                            'is_match': is_match
                        }
                        matches.append(match_data)
            
            # Sort matches by similarity score (highest first)
            matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Limit results
            matches = matches[:max_results]
            
            return {
                'success': True,
                'matches': matches,
                'total_matches': len(matches),
                'search_tolerance': self.tolerance
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'matches': [],
                'total_matches': 0
            }
    
    def search_by_face_id(self, detected_faces, selected_face_id, max_results=5):
        """
        Search using a specific face from detected faces
        
        Args:
            detected_faces (list): List of detected faces from face detector
            selected_face_id (int): ID of the selected face (1, 2, 3, etc.)
            max_results (int): Maximum number of results to return
            
        Returns:
            dict: Search results
        """
        try:
            # Find the selected face
            selected_face = None
            for face in detected_faces:
                if face['face_id'] == selected_face_id:
                    selected_face = face
                    break
            
            if not selected_face:
                return {
                    'success': False,
                    'error': f'Face {selected_face_id} not found in detected faces',
                    'matches': []
                }
            
            # Perform search
            return self.search_face(selected_face['encoding'], max_results)
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'matches': []
            }
    
    def add_profile_to_search(self, profile_data, face_encoding, face_image_path):
        """
        Add a new profile to the search database
        
        Args:
            profile_data (dict): Profile information
            face_encoding (numpy.ndarray): Face encoding
            face_image_path (str): Path to face image
            
        Returns:
            str or None: Profile ID if successful, None otherwise
        """
        return self.database.add_profile(profile_data, face_encoding, face_image_path)
    
    def get_all_profiles(self):
        """Get all profiles from the database"""
        return self.database.get_all_profiles()
    
    def get_profile(self, profile_id):
        """Get a specific profile"""
        return self.database.get_profile(profile_id)
    
    def delete_profile(self, profile_id):
        """Delete a profile from the database"""
        return self.database.delete_profile(profile_id)
    
    def update_tolerance(self, new_tolerance):
        """
        Update the face matching tolerance
        
        Args:
            new_tolerance (float): New tolerance value
        """
        self.tolerance = max(0.0, min(1.0, new_tolerance))
    
    def get_search_statistics(self):
        """
        Get search engine statistics
        
        Returns:
            dict: Statistics about the search database
        """
        stats = self.database.get_database_stats()
        stats['current_tolerance'] = self.tolerance
        return stats
    
    def bulk_search(self, query_encodings, max_results_per_query=3):
        """
        Search multiple faces at once
        
        Args:
            query_encodings (list): List of face encodings to search
            max_results_per_query (int): Max results per query
            
        Returns:
            dict: Results for all queries
        """
        try:
            results = []
            
            for i, encoding in enumerate(query_encodings):
                query_result = self.search_face(encoding, max_results_per_query)
                query_result['query_id'] = i + 1
                results.append(query_result)
            
            return {
                'success': True,
                'total_queries': len(query_encodings),
                'results': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'results': []
            }
    
    def find_similar_faces(self, profile_id, max_results=5):
        """
        Find faces similar to a specific profile in the database
        
        Args:
            profile_id (str): Profile ID to find similar faces for
            max_results (int): Maximum results to return
            
        Returns:
            dict: Similar faces
        """
        try:
            # Get the target profile's encoding
            target_encoding = self.database.get_face_encoding(profile_id)
            
            if target_encoding is None:
                return {
                    'success': False,
                    'error': 'Profile not found',
                    'matches': []
                }
            
            # Search for similar faces (excluding the target profile)
            search_result = self.search_face(target_encoding, max_results + 1)
            
            if search_result['success']:
                # Remove the target profile from results
                matches = [match for match in search_result['matches'] 
                          if match['profile_id'] != profile_id]
                
                return {
                    'success': True,
                    'target_profile_id': profile_id,
                    'matches': matches[:max_results],
                    'total_matches': len(matches)
                }
            else:
                return search_result
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'matches': []
            }
