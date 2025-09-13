import json
import numpy as np
import os
from datetime import datetime
import pickle

class FaceDatabase:
    def __init__(self, database_path="database"):
        """
        Initialize the face database manager
        
        Args:
            database_path (str): Path to the database directory
        """
        self.database_path = database_path
        self.profiles_file = os.path.join(database_path, "profiles.json")
        self.encodings_file = os.path.join(database_path, "face_encodings.pkl")
        
        # Create database directory if it doesn't exist
        os.makedirs(database_path, exist_ok=True)
        
        # Initialize database files if they don't exist
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database files if they don't exist"""
        if not os.path.exists(self.profiles_file):
            with open(self.profiles_file, 'w') as f:
                json.dump({}, f)
        
        if not os.path.exists(self.encodings_file):
            with open(self.encodings_file, 'wb') as f:
                pickle.dump({}, f)
    
    def add_profile(self, profile_data, face_encoding, face_image_path):
        """
        Add a new profile to the database
        
        Args:
            profile_data (dict): Profile information (name, age, etc.)
            face_encoding (numpy.ndarray): Face encoding
            face_image_path (str): Path to the face image
            
        Returns:
            str: Profile ID if successful, None otherwise
        """
        try:
            # Generate unique profile ID
            profile_id = self._generate_profile_id()
            
            # Load existing profiles
            with open(self.profiles_file, 'r') as f:
                profiles = json.load(f)
            
            # Add profile data
            profile_info = {
                'id': profile_id,
                'name': profile_data.get('name', 'Unknown'),
                'age': profile_data.get('age', ''),
                'description': profile_data.get('description', ''),
                'image_path': face_image_path,
                'created_date': datetime.now().isoformat(),
                'metadata': profile_data.get('metadata', {})
            }
            
            profiles[profile_id] = profile_info
            
            # Save updated profiles
            with open(self.profiles_file, 'w') as f:
                json.dump(profiles, f, indent=2)
            
            # Load existing encodings
            with open(self.encodings_file, 'rb') as f:
                encodings = pickle.load(f)
            
            # Add face encoding
            encodings[profile_id] = face_encoding
            
            # Save updated encodings
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(encodings, f)
            
            return profile_id
            
        except Exception as e:
            print(f"Error adding profile: {str(e)}")
            return None
    
    def get_all_profiles(self):
        """
        Get all profiles from the database
        
        Returns:
            dict: Dictionary of all profiles
        """
        try:
            with open(self.profiles_file, 'r') as f:
                profiles = json.load(f)
            return profiles
        except Exception as e:
            print(f"Error getting profiles: {str(e)}")
            return {}
    
    def get_profile(self, profile_id):
        """
        Get a specific profile by ID
        
        Args:
            profile_id (str): Profile ID
            
        Returns:
            dict or None: Profile data if found, None otherwise
        """
        try:
            with open(self.profiles_file, 'r') as f:
                profiles = json.load(f)
            return profiles.get(profile_id)
        except Exception as e:
            print(f"Error getting profile: {str(e)}")
            return None
    
    def get_all_face_encodings(self):
        """
        Get all face encodings from the database
        
        Returns:
            dict: Dictionary of profile_id -> face_encoding
        """
        try:
            with open(self.encodings_file, 'rb') as f:
                encodings = pickle.load(f)
            return encodings
        except Exception as e:
            print(f"Error getting encodings: {str(e)}")
            return {}
    
    def get_face_encoding(self, profile_id):
        """
        Get face encoding for a specific profile
        
        Args:
            profile_id (str): Profile ID
            
        Returns:
            numpy.ndarray or None: Face encoding if found, None otherwise
        """
        try:
            with open(self.encodings_file, 'rb') as f:
                encodings = pickle.load(f)
            return encodings.get(profile_id)
        except Exception as e:
            print(f"Error getting encoding: {str(e)}")
            return None
    
    def delete_profile(self, profile_id):
        """
        Delete a profile from the database
        
        Args:
            profile_id (str): Profile ID to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Remove from profiles
            with open(self.profiles_file, 'r') as f:
                profiles = json.load(f)
            
            if profile_id in profiles:
                # Delete associated image file
                image_path = profiles[profile_id].get('image_path')
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)
                
                del profiles[profile_id]
                
                with open(self.profiles_file, 'w') as f:
                    json.dump(profiles, f, indent=2)
            
            # Remove from encodings
            with open(self.encodings_file, 'rb') as f:
                encodings = pickle.load(f)
            
            if profile_id in encodings:
                del encodings[profile_id]
                
                with open(self.encodings_file, 'wb') as f:
                    pickle.dump(encodings, f)
            
            return True
            
        except Exception as e:
            print(f"Error deleting profile: {str(e)}")
            return False
    
    def update_profile(self, profile_id, updated_data):
        """
        Update profile information
        
        Args:
            profile_id (str): Profile ID to update
            updated_data (dict): Updated profile data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(self.profiles_file, 'r') as f:
                profiles = json.load(f)
            
            if profile_id in profiles:
                # Update fields
                for key, value in updated_data.items():
                    if key != 'id':  # Don't allow ID changes
                        profiles[profile_id][key] = value
                
                profiles[profile_id]['updated_date'] = datetime.now().isoformat()
                
                with open(self.profiles_file, 'w') as f:
                    json.dump(profiles, f, indent=2)
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Error updating profile: {str(e)}")
            return False
    
    def _generate_profile_id(self):
        """Generate a unique profile ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"profile_{timestamp}"
    
    def get_database_stats(self):
        """
        Get database statistics
        
        Returns:
            dict: Database statistics
        """
        try:
            profiles = self.get_all_profiles()
            encodings = self.get_all_face_encodings()
            
            return {
                'total_profiles': len(profiles),
                'total_encodings': len(encodings),
                'database_size': self._get_directory_size(self.database_path)
            }
            
        except Exception as e:
            print(f"Error getting stats: {str(e)}")
            return {'total_profiles': 0, 'total_encodings': 0, 'database_size': 0}
    
    def _get_directory_size(self, directory):
        """Get total size of directory in bytes"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
