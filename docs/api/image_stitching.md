# Image Stitching API Documentation

## Overview
This module provides functionality for creating panoramas and stitching multiple images together.

## Modules

### panorama_generator.py
Functions for creating panoramic images from multiple input images.

#### `create_panorama(images: List[np.ndarray]) -> np.ndarray`
Creates a panorama from a list of overlapping images.

Parameters:
- images: List of input images in sequence

Returns:
- Stitched panorama image

### image_aligner.py
Functions for aligning and warping images.

#### `align_images(image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`
Aligns two images using feature matching and homography.

Parameters:
- image1: First input image
- image2: Second input image

Returns:
- Tuple of aligned images

### blender.py
Functions for blending overlapping image regions.

#### `blend_images(image1: np.ndarray, image2: np.ndarray, mask: np.ndarray) -> np.ndarray`
Blends two images using specified mask.

Parameters:
- image1: First input image
- image2: Second input image
- mask: Blending mask

Returns:
- Blended image
