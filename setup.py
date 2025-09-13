#!/usr/bin/env python3
"""
Face Search Package Setup Script

This setup script configures the package for pip installation and provides
setup utilities for the face search system.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read version from package
sys.path.insert(0, str(this_directory / "face_search_package"))
from face_search_package import __version__, __author__, __email__

# Define package requirements
requirements = [
    "face-recognition>=1.3.0",
    "opencv-python>=4.8.0",
    "Pillow>=10.0.0",
    "numpy>=1.24.0",
    "flask>=2.3.0",  # For demo web app only
    "werkzeug>=2.3.0",  # For demo web app only
]

# Additional requirements for demo web application
demo_requirements = [
    "flask>=2.3.0",
    "werkzeug>=2.3.0",
]

# Development requirements
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]

setup(
    name="face-search-package",
    version=__version__,
    author=__author__,
    author_email=__email__,
    description="A comprehensive face recognition and search system with advanced AI capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harshnath628/face-search-system",  # Update with your GitHub URL
    project_urls={
        "Bug Reports": "https://github.com/harshnath628/face-search-system/issues",
        "Source": "https://github.com/harshnath628/face-search-system",
        "Documentation": "https://github.com/harshnath628/face-search-system#readme",
    },
    packages=find_packages(include=['face_search_package', 'face_search_package.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "demo": demo_requirements,
        "dev": dev_requirements,
        "all": demo_requirements + dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "face-search-demo=demo_webapp.app:main",  # Optional demo command
        ],
    },
    include_package_data=True,
    package_data={
        "face_search_package": ["*.py"],
    },
    keywords=[
        "face-recognition", "computer-vision", "artificial-intelligence",
        "opencv", "machine-learning", "face-detection", "image-processing",
        "deep-learning", "neural-networks", "biometrics"
    ],
    zip_safe=False,  # Required for proper package loading
)


# Setup utilities (when run as script)
def setup_demo_environment():
    """Set up demo environment"""
    import subprocess
    
    print("Setting up demo environment...")
    
    try:
        # Create directories
        directories = ["uploads", "database", "static"]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            print(f"✓ Created directory: {directory}")
        
        # Initialize database
        from face_search_package import FaceSearchSystem
        face_system = FaceSearchSystem()
        print("✓ Face database initialized!")
        
        print("\n" + "=" * 50)
        print("✓ Demo environment setup completed!")
        print("=" * 50)
        print("\nTo run the demo web application:")
        print("python demo_webapp/app.py")
        print("\nOr use the package in your code:")
        print("from face_search_package import FaceSearchSystem")
        print("face_system = FaceSearchSystem()")
        
    except Exception as e:
        print(f"ERROR: Setup failed: {e}")
        sys.exit(1)


def check_installation():
    """Check if the package is properly installed"""
    try:
        from face_search_package import FaceSearchSystem, Config
        print("✓ Face Search Package is properly installed")
        
        # Test basic functionality
        config = Config()
        face_system = FaceSearchSystem(config)
        stats = face_system.get_statistics()
        
        print(f"✓ System initialized with {stats.get('total_profiles', 0)} profiles")
        print("✓ All components working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Installation check failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Search Package Setup Utilities")
    parser.add_argument("--setup-demo", action="store_true", help="Set up demo environment")
    parser.add_argument("--check", action="store_true", help="Check installation")
    
    args = parser.parse_args()
    
    if args.setup_demo:
        setup_demo_environment()
    elif args.check:
        if check_installation():
            print("\n🎉 Everything is working correctly!")
        else:
            print("\n❌ Installation issues detected")
            sys.exit(1)
    else:
        print("Face Search Package Setup")
        print("\nUsage:")
        print("  pip install -e .              # Install package in development mode")
        print("  python setup.py --setup-demo  # Set up demo environment")
        print("  python setup.py --check       # Check installation")
        print("\nFor more information, see README.md")
