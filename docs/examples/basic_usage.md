# Basic Usage Examples

## Video Frame Extraction

```python
from opencv_toolkit.image_processing import video_converter

# Extract frames from a video
video_path = "input/video.mp4"
output_dir = "output/frames"
frames = video_converter.extract_frames(video_path, output_dir, fps=30)
print(f"Extracted {len(frames)} frames")
```

## Feature Detection and Matching

```python
from opencv_toolkit.feature_analysis import feature_matcher
import cv2

# Load images
image1 = cv2.imread("image1.jpg")
image2 = cv2.imread("image2.jpg")

# Detect and match features
keypoints1, descriptors1 = feature_matcher.detect_features(image1)
keypoints2, descriptors2 = feature_matcher.detect_features(image2)
matches = feature_matcher.match_features(descriptors1, descriptors2)

print(f"Found {len(matches)} matches")
```

## Creating Panoramas

```python
from opencv_toolkit.image_stitching import panorama_generator
import cv2

# Load sequence of images
images = [
    cv2.imread(f"image{i}.jpg") 
    for i in range(1, 4)
]

# Create panorama
panorama = panorama_generator.create_panorama(images)
cv2.imwrite("panorama.jpg", panorama)
```
