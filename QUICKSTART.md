# Quick Start Guide - Face Search Package

Get up and running with the Face Search Package in minutes!

## 🚀 Installation

```bash
# Install the package
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

## 💡 Basic Usage (5 lines of code!)

```python
from face_search_package import FaceSearchSystem

# Initialize the system
face_system = FaceSearchSystem()

# Add a profile
profile_id = face_system.add_profile(
    name="John Doe",
    image_path="path/to/john.jpg"
)

# Search for faces in an image
results = face_system.search_and_match("group_photo.jpg")

# Print results
for face_result in results['face_results']:
    for match in face_result['matches']:
        print(f"{match['profile']['name']}: {match['similarity_score']}% match")
```

## 🔧 Configuration Options

```python
from face_search_package import FaceSearchSystem, ConfigPresets

# High accuracy mode
face_system = FaceSearchSystem(ConfigPresets.high_accuracy())

# Custom configuration
config = {
    'database_path': 'my_faces',
    'default_tolerance': 0.5,
    'similarity_threshold': 80.0
}
face_system = FaceSearchSystem(config)
```

## 🌐 Web Demo

```bash
# Run the demo web application
cd demo_webapp
python app.py

# Open http://localhost:5000 in your browser
```

## 📚 Integration Examples

### Flask Web App
```python
from face_search_package import FaceSearchSystem

app = Flask(__name__)
face_system = FaceSearchSystem()

@app.route('/search', methods=['POST'])
def search():
    file = request.files['image']
    temp_path = save_temp_file(file)
    results = face_system.search_and_match(temp_path)
    return jsonify(results)
```

### Batch Processing
```python
from face_search_package import FaceSearchSystem, ConfigPresets

# Memory efficient for large batches
face_system = FaceSearchSystem(ConfigPresets.memory_efficient())

for image_path in image_list:
    results = face_system.search_and_match(image_path)
    process_results(results)
```

### Desktop Application
```python
import tkinter as tk
from face_search_package import FaceSearchSystem

class FaceSearchApp:
    def __init__(self):
        self.face_system = FaceSearchSystem()
        self.setup_ui()
    
    def search_faces(self):
        results = self.face_system.search_and_match(self.image_path)
        self.display_results(results)
```

## 🎯 Key Features

- **One-line face search**: `face_system.search_and_match("image.jpg")`
- **Multiple configuration presets**: High accuracy, fast processing, memory efficient
- **Flexible API**: Use individual components or the high-level wrapper
- **Production ready**: Proper error handling, logging, and validation
- **Extensible**: Easy to customize and integrate

## 📖 Next Steps

- [View Examples](examples/) - Complete integration examples
- [Read Documentation](README.md) - Full documentation with theory
- [Run Demo](demo_webapp/) - Interactive web demonstration

## 🆘 Need Help?

- Check the [Integration Guide](examples/integration_guide.py)
- Look at [Basic Usage Examples](examples/basic_usage.py)
- Review the comprehensive [README](README.md)

---

**That's it!** You're ready to add face recognition to your project. The package handles all the complexity while giving you a simple, powerful API.
