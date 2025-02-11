"""
HSV and Sliding Window Lane Detection Module.

This module provides functionality for lane detection using HSV color space filtering
and sliding window search technique. It includes functions for image preprocessing,
perspective transformation, and histogram analysis.

Original source:
https://github.com/SHAHFAISAL80/detect-curved-lane-lines-using-HSV-filtering-and-sliding-window-search
"""

import logging
from typing import Tuple, Optional, Union
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure matplotlib backend
try:
    matplotlib.use('TkAgg')  # Use 'Agg' for non-GUI environments
except Exception as e:
    logger.warning(f"Failed to set matplotlib backend: {e}")


def pipeline(
    img: np.ndarray,
    s_thresh: Tuple[int, int] = (100, 255),
    sx_thresh: Tuple[int, int] = (15, 255)
) -> np.ndarray:
    """
    Process an image through a lane detection pipeline using HLS color space and Sobel filters.

    Args:
        img: Input image in RGB format
        s_thresh: Saturation channel threshold values (min, max)
        sx_thresh: Sobel x gradient threshold values (min, max)

    Returns:
        Binary image with detected lane lines
    """
    try:
        img = np.copy(img)
        # Convert to HLS color space and separate the channels
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float64)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # Apply Sobel x gradient
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Create binary images based on thresholds
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # Combine binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return combined_binary

    except Exception as e:
        logger.error(f"Error in pipeline processing: {e}")
        raise


def perspective_warp(
    img: np.ndarray,
    dst_size: Tuple[int, int] = (1280, 720),
    src: np.ndarray = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]),
    dst: np.ndarray = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])
) -> np.ndarray:
    """
    Apply perspective transformation to an image.

    Args:
        img: Input image
        dst_size: Output image size (width, height)
        src: Source points for perspective transform (normalized coordinates)
        dst: Destination points for perspective transform (normalized coordinates)

    Returns:
        Warped image with bird's eye view perspective
    """
    try:
        img_size = np.float32([(img.shape[1], img.shape[0])])
        src = src * img_size
        dst = dst * np.float32(dst_size)

        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, dst_size)

        return warped

    except Exception as e:
        logger.error(f"Error in perspective warping: {e}")
        raise


def inv_perspective_warp(
    img: np.ndarray,
    dst_size: Tuple[int, int] = (1280, 720),
    src: np.ndarray = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]),
    dst: np.ndarray = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])
) -> np.ndarray:
    """
    Apply inverse perspective transformation to return to original view.

    Args:
        img: Input warped image
        dst_size: Output image size (width, height)
        src: Source points for inverse transform (normalized coordinates)
        dst: Destination points for inverse transform (normalized coordinates)

    Returns:
        Unwarped image in original perspective
    """
    try:
        img_size = np.float32([(img.shape[1], img.shape[0])])
        src = src * img_size
        dst = dst * np.float32(dst_size)

        # Calculate inverse perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        unwarped = cv2.warpPerspective(img, M, dst_size)

        return unwarped

    except Exception as e:
        logger.error(f"Error in inverse perspective warping: {e}")
        raise


def get_hist(img: np.ndarray) -> np.ndarray:
    """
    Calculate histogram of the lower half of the binary image.

    Args:
        img: Binary image

    Returns:
        Histogram array of pixel counts
    """
    try:
        return np.sum(img[img.shape[0] // 2:, :], axis=0)
    except Exception as e:
        logger.error(f"Error calculating histogram: {e}")
        raise


def process_image(image_path: str) -> None:
    """
    Process and visualize lane detection on a single image.

    Args:
        image_path: Path to the input image file
    """
    try:
        # Read and convert image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply pipeline and perspective transform
        dst = pipeline(img)
        dst = perspective_warp(dst, dst_size=(1280, 720))

        # Visualization
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(dst, cmap='gray')
        ax2.set_title('Warped Image', fontsize=30)
        
        plt.show()

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise


if __name__ == "__main__":
    try:
        process_image('C:/src/opencv/extracted_frames_01/frame_00457.jpg')
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
