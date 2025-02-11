# Getting Started with OpenCV Toolkit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/opencv-toolkit.git
cd opencv-toolkit
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your settings:
```env
DEBUG=True
LOG_LEVEL=DEBUG
OUTPUT_DIR=./data/output
```

## Project Structure

```
opencv_toolkit/
├── src/                    # Source code
│   ├── image_processing/   # Image preprocessing
│   ├── feature_analysis/   # Feature detection
│   ├── image_stitching/    # Panorama creation
│   └── common/            # Shared utilities
├── tests/                 # Test suite
├── docs/                 # Documentation
└── config/              # Configuration
```

## Quick Start

```python
from opencv_toolkit.image_processing import video_converter
from opencv_toolkit.feature_analysis import feature_matcher

# Extract frames from video
frames = video_converter.extract_frames("video.mp4", "output/frames")

# Detect features
for frame in frames:
    keypoints, descriptors = feature_matcher.detect_features(frame)
```

## Next Steps

- Check out the [Basic Usage Examples](../examples/basic_usage.md)
- Read the [API Documentation](../api/)
- Review [Configuration Options](../config/)
