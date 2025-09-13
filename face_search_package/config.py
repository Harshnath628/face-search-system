"""
Configuration management for Face Search Package.

Provides centralized configuration with defaults and easy customization.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """
    Configuration class for Face Search Package.
    
    Allows easy customization of all system parameters including paths,
    model settings, and search parameters.
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        # Database settings
        'database_path': 'face_search_data',
        'profiles_filename': 'profiles.json',
        'encodings_filename': 'face_encodings.pkl',
        
        # Search settings
        'default_tolerance': 0.6,
        'max_search_results': 10,
        'similarity_threshold': 60.0,  # Percentage
        
        # Face detection settings
        'face_detection_model': 'hog',  # 'hog' or 'cnn'
        'face_detection_upsample': 1,
        'min_face_size': (50, 50),
        
        # Image processing settings
        'max_image_size': (1024, 1024),
        'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.gif'],
        'image_quality': 95,
        
        # Performance settings
        'enable_parallel_processing': True,
        'max_workers': 4,
        'cache_encodings': True,
        
        # Security settings
        'secure_deletion': True,
        'encrypt_database': False,
        'log_searches': True,
        
        # Web interface settings (for demo app)
        'web_host': '127.0.0.1',
        'web_port': 5000,
        'web_debug': True,
        'max_upload_size': 16 * 1024 * 1024,  # 16MB
    }
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Dictionary of configuration overrides
            config_file: Path to JSON configuration file
        """
        self._config = self.DEFAULT_CONFIG.copy()
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
        
        # Apply dictionary overrides
        if config_dict:
            self._config.update(config_dict)
        
        # Ensure paths exist
        self._ensure_paths()
    
    def _load_from_file(self, config_file: str):
        """Load configuration from JSON file."""
        import json
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self._config.update(file_config)
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def _ensure_paths(self):
        """Ensure required directories exist."""
        database_path = Path(self.database_path)
        database_path.mkdir(parents=True, exist_ok=True)
    
    def __getattr__(self, name: str) -> Any:
        """Allow dot notation access to configuration values."""
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"Configuration has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration values."""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style setting of configuration values."""
        self._config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self._config.get(key, default)
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values."""
        self._config.update(updates)
        self._ensure_paths()
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def save_to_file(self, filepath: str):
        """Save current configuration to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self._config, f, indent=2, default=str)
    
    @property
    def database_full_path(self) -> str:
        """Get full path to database directory."""
        return os.path.abspath(self.database_path)
    
    @property
    def profiles_path(self) -> str:
        """Get full path to profiles file."""
        return os.path.join(self.database_path, self.profiles_filename)
    
    @property
    def encodings_path(self) -> str:
        """Get full path to encodings file."""
        return os.path.join(self.database_path, self.encodings_filename)
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate current configuration.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate tolerance
        if not 0.0 <= self.default_tolerance <= 1.0:
            errors.append("default_tolerance must be between 0.0 and 1.0")
        
        # Validate similarity threshold
        if not 0.0 <= self.similarity_threshold <= 100.0:
            errors.append("similarity_threshold must be between 0.0 and 100.0")
        
        # Validate face detection model
        if self.face_detection_model not in ['hog', 'cnn']:
            errors.append("face_detection_model must be 'hog' or 'cnn'")
        
        # Validate supported formats
        if not isinstance(self.supported_formats, list):
            errors.append("supported_formats must be a list")
        
        return len(errors) == 0, errors
    
    @classmethod
    def from_file(cls, filepath: str) -> 'Config':
        """Create Config instance from JSON file."""
        return cls(config_file=filepath)
    
    @classmethod
    def create_default_config_file(cls, filepath: str):
        """Create a default configuration file."""
        config = cls()
        config.save_to_file(filepath)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(database_path='{self.database_path}', tolerance={self.default_tolerance})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Config({self._config})"


# Predefined configuration presets
class ConfigPresets:
    """Predefined configuration presets for common use cases."""
    
    @staticmethod
    def high_accuracy() -> Dict[str, Any]:
        """Configuration optimized for high accuracy."""
        return {
            'default_tolerance': 0.4,
            'face_detection_model': 'cnn',
            'face_detection_upsample': 2,
            'similarity_threshold': 75.0,
            'min_face_size': (100, 100)
        }
    
    @staticmethod
    def fast_processing() -> Dict[str, Any]:
        """Configuration optimized for speed."""
        return {
            'default_tolerance': 0.6,
            'face_detection_model': 'hog',
            'face_detection_upsample': 0,
            'max_image_size': (512, 512),
            'enable_parallel_processing': True,
            'max_workers': 8
        }
    
    @staticmethod
    def memory_efficient() -> Dict[str, Any]:
        """Configuration optimized for low memory usage."""
        return {
            'cache_encodings': False,
            'max_image_size': (256, 256),
            'enable_parallel_processing': False,
            'max_workers': 1
        }
    
    @staticmethod
    def production() -> Dict[str, Any]:
        """Configuration suitable for production deployment."""
        return {
            'web_debug': False,
            'log_searches': True,
            'secure_deletion': True,
            'encrypt_database': True,
            'default_tolerance': 0.5,
            'similarity_threshold': 70.0
        }
