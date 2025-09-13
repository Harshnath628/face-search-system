"""
Basic Usage Example - Face Search Package

This example demonstrates the most common use cases of the face search system:
1. Adding profiles to the database
2. Searching for faces in images
3. Getting matches and similarity scores

Run this script to see the basic functionality in action.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_search_package import FaceSearchSystem, Config, ConfigPresets


def basic_example():
    """Demonstrate basic face search functionality."""
    print("=" * 60)
    print("Face Search Package - Basic Usage Example")
    print("=" * 60)
    
    # 1. Initialize the system with default configuration
    print("\n1. Initializing Face Search System...")
    face_system = FaceSearchSystem()
    print(f"✓ System initialized: {face_system}")
    
    # Display initial statistics
    stats = face_system.get_statistics()
    print(f"Initial profiles in database: {stats.get('total_profiles', 0)}")
    
    # 2. Add some example profiles (you would replace with your own images)
    print("\n2. Adding sample profiles...")
    
    # Note: In a real scenario, you would have actual image files
    sample_profiles = [
        {
            "name": "Alice Johnson",
            "age": "28",
            "description": "Software Engineer",
            "metadata": {"department": "Engineering", "employee_id": "ENG001"}
        },
        {
            "name": "Bob Smith", 
            "age": "35",
            "description": "Project Manager",
            "metadata": {"department": "Management", "employee_id": "MGR001"}
        }
    ]
    
    print("Note: This example assumes you have image files in an 'images' folder.")
    print("Replace the paths below with your actual image files.")
    
    for profile in sample_profiles:
        # In practice, you would have real image paths like:
        # image_path = f"images/{profile['name'].lower().replace(' ', '_')}.jpg"
        print(f"  - Would add profile: {profile['name']}")
        print(f"    Image path: images/{profile['name'].lower().replace(' ', '_')}.jpg")
        print(f"    Metadata: {profile['metadata']}")
    
    print("\n   Example code to add a profile:")
    print("   profile_id = face_system.add_profile(")
    print("       name='Alice Johnson',")
    print("       image_path='images/alice_johnson.jpg',")
    print("       age='28',")
    print("       description='Software Engineer',")
    print("       metadata={'department': 'Engineering'}")
    print("   )")
    
    # 3. Search for faces in an image
    print("\n3. Searching for faces in an image...")
    print("   Example code to search:")
    print("   results = face_system.search_faces_in_image('group_photo.jpg')")
    print("   print(f'Found {results[\"total_faces\"]} faces')")
    print("   ")
    print("   for face in results['faces']:")
    print("       print(f'Face {face[\"face_id\"]} at {face[\"coordinates\"]}')")
    
    # 4. Get matches for a specific face
    print("\n4. Finding matches for detected faces...")
    print("   Example code to get matches:")
    print("   matches = face_system.get_matches(")
    print("       face_encoding=detected_face['encoding'],")
    print("       max_results=5,")
    print("       min_similarity=70.0")
    print("   )")
    print("   ")
    print("   for match in matches['matches']:")
    print("       name = match['profile']['name']")
    print("       similarity = match['similarity_score']")
    print("       print(f'{name}: {similarity}% match')")
    
    # 5. One-step search and match
    print("\n5. One-step search and match...")
    print("   Example code for complete workflow:")
    print("   results = face_system.search_and_match(")
    print("       'group_photo.jpg',")
    print("       max_results=3,")
    print("       min_similarity=75.0")
    print("   )")
    print("   ")
    print("   for face_result in results['face_results']:")
    print("       print(f'Face {face_result[\"face_id\"]}:')")
    print("       for match in face_result['matches']:")
    print("           print(f'  - {match[\"profile\"][\"name\"]}: {match[\"similarity_score\"]}%')")
    
    print("\n" + "=" * 60)
    print("Basic example completed!")
    print("=" * 60)


def configuration_example():
    """Demonstrate different configuration options."""
    print("\n" + "=" * 60)
    print("Configuration Examples")
    print("=" * 60)
    
    # 1. Default configuration
    print("\n1. Default Configuration:")
    config = Config()
    print(f"   Database path: {config.database_path}")
    print(f"   Default tolerance: {config.default_tolerance}")
    print(f"   Max results: {config.max_search_results}")
    
    # 2. Custom configuration with dictionary
    print("\n2. Custom Configuration (Dictionary):")
    custom_config = {
        'database_path': 'my_face_database',
        'default_tolerance': 0.5,
        'max_search_results': 20,
        'similarity_threshold': 75.0
    }
    
    face_system = FaceSearchSystem(custom_config)
    print(f"   Database path: {face_system.config.database_path}")
    print(f"   Tolerance: {face_system.config.default_tolerance}")
    print(f"   Max results: {face_system.config.max_search_results}")
    
    # 3. Preset configurations
    print("\n3. Preset Configurations:")
    
    presets = {
        "High Accuracy": ConfigPresets.high_accuracy(),
        "Fast Processing": ConfigPresets.fast_processing(),
        "Memory Efficient": ConfigPresets.memory_efficient(),
        "Production": ConfigPresets.production()
    }
    
    for name, preset in presets.items():
        print(f"\n   {name}:")
        print(f"     Tolerance: {preset.get('default_tolerance', 'default')}")
        print(f"     Detection Model: {preset.get('face_detection_model', 'default')}")
        print(f"     Similarity Threshold: {preset.get('similarity_threshold', 'default')}%")
    
    # Example of using a preset
    print("\n   Using high accuracy preset:")
    print("   face_system = FaceSearchSystem(ConfigPresets.high_accuracy())")


def context_manager_example():
    """Demonstrate context manager usage."""
    print("\n" + "=" * 60)
    print("Context Manager Example")
    print("=" * 60)
    
    print("\nUsing the system as a context manager:")
    print("with FaceSearchSystem() as face_system:")
    print("    # System is automatically cleaned up when done")
    print("    stats = face_system.get_statistics()")
    print("    print(f'System has {stats.get(\"total_profiles\", 0)} profiles')")
    
    # Actual demonstration
    with FaceSearchSystem() as face_system:
        stats = face_system.get_statistics()
        print(f"\n✓ System has {stats.get('total_profiles', 0)} profiles")
        print("✓ Context manager working correctly")


def validation_example():
    """Demonstrate image validation."""
    print("\n" + "=" * 60)
    print("Image Validation Example")
    print("=" * 60)
    
    face_system = FaceSearchSystem()
    
    # Test validation with various scenarios
    test_cases = [
        "nonexistent_file.jpg",  # File doesn't exist
        "README.md",  # Wrong file type (if it exists)
        # You could add more test cases with real files
    ]
    
    for test_file in test_cases:
        is_valid, error = face_system.validate_image(test_file)
        status = "✓ Valid" if is_valid else f"✗ Invalid: {error}"
        print(f"   {test_file}: {status}")
    
    print("\nValidation checks:")
    print("   - File existence")
    print("   - File format (jpg, png, gif, bmp)")
    print("   - File size limits")
    print("   - Image corruption")
    print("   - Minimum image dimensions")


def main():
    """Run all examples."""
    try:
        basic_example()
        configuration_example()
        context_manager_example()
        validation_example()
        
        print("\n🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("1. Create an 'images' folder with sample photos")
        print("2. Run the examples with real image files")
        print("3. Explore the advanced examples")
        print("4. Integration the package into your own project")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure the face_search_package is properly installed")


if __name__ == "__main__":
    main()
