# Video Processing Tutorial

This tutorial demonstrates how to use the OpenCV Toolkit for video processing tasks.

## 1. Video Frame Extraction

### Basic Frame Extraction
```python
from opencv_toolkit.image_processing import video_converter

# Extract frames at default rate (30 fps)
frames = video_converter.extract_frames(
    video_path="input.mp4",
    output_dir="output/frames"
)
```

### Custom Frame Rate
```python
# Extract frames at 10 fps
frames = video_converter.extract_frames(
    video_path="input.mp4",
    output_dir="output/frames",
    fps=10
)
```

## 2. Frame Processing

### Remove Black Borders
```python
from opencv_toolkit.image_processing import image_cleaner

# Clean a single frame
cleaned_frame = image_cleaner.remove_black_areas(frame)

# Process multiple frames
cleaned_frames = [
    image_cleaner.remove_black_areas(frame)
    for frame in frames
]
```

### Color Space Processing
```python
from opencv_toolkit.image_processing import color_processor

# Convert to HSV
hsv_frame = color_processor.convert_to_hsv(frame)

# Apply color filtering
mask = color_processor.create_color_mask(
    hsv_frame,
    lower_bound=(0, 50, 50),
    upper_bound=(10, 255, 255)
)
```

## 3. Feature Detection in Video

```python
from opencv_toolkit.feature_analysis import feature_matcher

# Process sequence of frames
for frame in frames:
    # Detect features
    keypoints, descriptors = feature_matcher.detect_features(frame)
    
    # Match with template
    matches = feature_matcher.match_features(
        descriptors,
        template_descriptors
    )
```

## 4. Saving Results

```python
import cv2

# Save processed frames
for i, frame in enumerate(processed_frames):
    cv2.imwrite(f"output/processed_{i:04d}.jpg", frame)

# Create video from frames
video_converter.frames_to_video(
    frame_dir="output/processed",
    output_path="output/result.mp4",
    fps=30
)
```
