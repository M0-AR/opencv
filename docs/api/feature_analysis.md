# Feature Analysis API Documentation

## Overview
This module provides functionality for detecting and analyzing features in images, including feature matching, similarity calculation, and template matching.

## Modules

### feature_matcher.py
Functions for detecting and matching features between images.

#### `detect_features(image: np.ndarray, method: str = 'SIFT') -> Tuple[List[cv2.KeyPoint], np.ndarray]`
Detects features in an image using specified method.

Parameters:
- image: Input image
- method: Feature detection method ('SIFT', 'SURF', 'ORB')

Returns:
- Tuple of (keypoints, descriptors)

### image_similarity.py
Functions for calculating similarity between images.

#### `calculate_similarity(image1: np.ndarray, image2: np.ndarray) -> float`
Calculates similarity score between two images.

Parameters:
- image1: First input image
- image2: Second input image

Returns:
- Similarity score (0-1)

### template_matcher.py
Functions for template matching and object detection.

#### `find_template(image: np.ndarray, template: np.ndarray) -> List[Tuple[int, int, int, int]]`
Finds occurrences of template in image.

Parameters:
- image: Source image to search in
- template: Template to find

Returns:
- List of bounding boxes (x, y, w, h)
