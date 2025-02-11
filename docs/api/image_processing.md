# Image Processing API Documentation

## Overview
This module provides functionality for basic image processing operations including video frame extraction, image cleaning, and color space manipulations.

## Modules

### video_converter.py
Functions for converting video files to image sequences.

#### `extract_frames(video_path: str, output_dir: str, fps: int = 30) -> List[str]`
Extracts frames from a video file at specified intervals.

Parameters:
- video_path: Path to the input video file
- output_dir: Directory to save extracted frames
- fps: Frames per second to extract (default: 30)

Returns:
- List of paths to extracted frame images

### image_cleaner.py
Functions for cleaning and preprocessing images.

#### `remove_black_areas(image: np.ndarray) -> np.ndarray`
Removes black borders and artifacts from images.

Parameters:
- image: Input image as numpy array

Returns:
- Cleaned image as numpy array

### color_processor.py
Color space manipulation and analysis functions.

#### `convert_to_hsv(image: np.ndarray) -> np.ndarray`
Converts RGB image to HSV color space.

Parameters:
- image: Input RGB image

Returns:
- HSV converted image
