"""Bilateral Filter Processing Module.

This module implements bilateral filtering for video streams, which effectively
reduces noise while preserving edges. The bilateral filter is non-linear, edge-preserving,
and noise-reducing smoothing filter for images.

Date: 2025-02-11
"""

from typing import Tuple, Optional
import logging
import sys

import cv2
import numpy as np
from numpy.typing import NDArray

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_video(video_path: str) -> Tuple[Optional[cv2.VideoCapture], bool]:
    """Load and validate a video file for processing.

    Args:
        video_path: Path to the input video file.

    Returns:
        Tuple containing:
            - VideoCapture object if successful, None otherwise
            - Boolean indicating success status

    Raises:
        ValueError: If video path is empty
        FileNotFoundError: If video file cannot be opened
    """
    if not video_path:
        raise ValueError("Video path cannot be empty")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")
        return cap, True
    except Exception as e:
        logger.error(f"Error loading video: {str(e)}")
        return None, False


def apply_bilateral_filter(
    frame: NDArray,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0
) -> Optional[NDArray]:
    """Apply bilateral filter to a frame.

    Args:
        frame: Input frame to process
        diameter: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space

    Returns:
        Processed frame with bilateral filter applied, or None if processing fails
    """
    try:
        return cv2.bilateralFilter(
            frame,
            diameter,
            sigma_color,
            sigma_space
        )
    except Exception as e:
        logger.error(f"Error applying bilateral filter: {str(e)}")
        return None


def create_comparison_view(
    original: NDArray,
    processed: NDArray,
    target_size: Tuple[int, int] = (1400, 700)
) -> Optional[NDArray]:
    """Create a side-by-side comparison view of original and processed frames.

    Args:
        original: Original frame
        processed: Processed frame
        target_size: Desired output resolution (width, height)

    Returns:
        Combined and resized view of both frames, or None if processing fails
    """
    try:
        combined = cv2.hconcat([original, processed])
        return cv2.resize(combined, target_size)
    except Exception as e:
        logger.error(f"Error creating comparison view: {str(e)}")
        return None


def process_video_stream(
    cap: cv2.VideoCapture,
    filter_params: dict = None
) -> None:
    """Process video stream with bilateral filter and display results.

    Args:
        cap: OpenCV VideoCapture object
        filter_params: Dictionary of bilateral filter parameters
    """
    if filter_params is None:
        filter_params = {
            'diameter': 9,
            'sigma_color': 75.0,
            'sigma_space': 75.0
        }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply bilateral filter
            filtered_frame = apply_bilateral_filter(
                frame,
                **filter_params
            )
            if filtered_frame is None:
                continue

            # Create comparison view
            comparison = create_comparison_view(frame, filtered_frame)
            if comparison is None:
                continue

            # Display results
            cv2.imshow('Original and Bilateral Filtered Frames', comparison)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main(video_path: str) -> int:
    """Main function to run bilateral filtering on video.

    Args:
        video_path: Path to the video file

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Initialize video capture
        cap, success = load_video(video_path)
        if not success:
            return 1

        # Process video stream
        process_video_stream(cap)
        return 0

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    video_path = '020146-3173 (35).mp4'
    sys.exit(main(video_path))
