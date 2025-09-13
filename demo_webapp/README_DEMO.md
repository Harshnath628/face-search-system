# Demo Web Application

This directory contains a complete web application demonstrating the Face Search Package functionality.

## Features

- Upload images and detect faces
- Add profiles to the database
- Search for matching faces
- View and manage all profiles
- Adjust search settings

## Running the Demo

1. **Install the package first:**
   ```bash
   pip install -e ..
   ```

2. **Run the demo web application:**
   ```bash
   python app.py
   ```

3. **Open your browser and go to:**
   ```
   http://localhost:5000
   ```

## Using the Demo

1. **Add Profiles**: Start by adding some profiles using "Add New Profile"
2. **Upload Search Image**: Use "Search by Image" to upload a photo with faces
3. **Select Face**: Choose which detected face to search for
4. **View Results**: See matching profiles with similarity scores

## Configuration

The demo uses the Face Search Package with default settings. You can modify the configuration in `app.py`:

```python
# Example: Use high accuracy settings
from face_search_package import ConfigPresets
config = ConfigPresets.high_accuracy()
face_system = FaceSearchSystem(config)
```

## Integration Example

This demo shows how to integrate the Face Search Package into a web application. Key integration points:

- **File Upload Handling**: Safe temporary file management
- **Error Handling**: User-friendly error messages
- **Session Management**: Multi-step face detection and search workflow
- **Results Display**: Professional UI for search results

## Files

- `app.py` - Main Flask application
- `templates/` - HTML templates with Bootstrap styling
- All templates include responsive design and modern UI components
