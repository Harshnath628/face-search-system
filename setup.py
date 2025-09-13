#!/usr/bin/env python3
"""
Face Search System Setup Script
This script helps install dependencies and set up the face search system.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7 or higher is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python version: {sys.version.split()[0]}")

def install_requirements():
    """Install required packages"""
    print("\nInstalling required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install packages: {e}")
        print("\nTry installing packages manually:")
        print("pip install flask opencv-python face-recognition Pillow numpy werkzeug")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    directories = ["uploads", "database", "static", "templates"]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(exist_ok=True)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory exists: {directory}")

def initialize_database():
    """Initialize the face database"""
    from face_database import FaceDatabase
    
    try:
        db = FaceDatabase()
        print("✓ Face database initialized successfully!")
    except Exception as e:
        print(f"ERROR: Failed to initialize database: {e}")
        sys.exit(1)

def check_dependencies():
    """Check if all critical dependencies are available"""
    critical_packages = ['cv2', 'face_recognition', 'flask', 'PIL', 'numpy']
    
    print("\nChecking dependencies...")
    for package in critical_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            print(f"✗ {package} is not available")
            return False
    return True

def main():
    """Main setup function"""
    print("=" * 50)
    print("Face Search System Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Install requirements
    install_requirements()
    
    # Check dependencies
    if not check_dependencies():
        print("\nERROR: Some dependencies are missing. Please check the installation.")
        sys.exit(1)
    
    # Initialize database
    print("\nInitializing database...")
    initialize_database()
    
    print("\n" + "=" * 50)
    print("✓ Setup completed successfully!")
    print("=" * 50)
    print("\nTo start the application:")
    print("python app.py")
    print("\nThen open your browser and go to:")
    print("http://localhost:5000")
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
