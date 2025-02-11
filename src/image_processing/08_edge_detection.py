"""Edge Detection and Contour Analysis Module.

This module provides functionality for edge detection, contour analysis, and
Hu moment calculation in video streams. It combines multiple image processing
techniques to detect and analyze shapes in video frames.

Features:
- Video stream processing
- Edge detection using Canny algorithm
- Contour detection and analysis
- Hu moment calculation
- Real-time visualization

Date: 2025-02-11
"""

from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import logging
import sys
import os

import cv2
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VideoProcessingConfig:
    """Configuration parameters for video processing."""
    edge_low_threshold: int = 50
    edge_high_threshold: int = 150
    min_contour_area: float = 100.0
    visualization_size: Tuple[int, int] = (1400, 700)
    display_scale: float = 1.0


@dataclass
class ContourAnalysisResult:
    """Results from contour analysis."""
    contour_image: NDArray
    hu_moments: List[float]
    contour_count: int
    total_area: float


def load_video(video_path: str) -> Tuple[Optional[cv2.VideoCapture], Dict[str, float]]:
    """Load and validate a video file for processing.

    Args:
        video_path: Path to the input video file.

    Returns:
        Tuple containing:
            - VideoCapture object if successful, None otherwise
            - Dictionary with video properties (fps, frame_count, duration)

    Raises:
        ValueError: If video path is empty or invalid
        FileNotFoundError: If video file cannot be opened
        RuntimeError: If video properties cannot be read
    """
    if not video_path:
        raise ValueError("Video path cannot be empty")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        properties = {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration
        }

        return cap, properties
    except Exception as e:
        logger.error(f"Error loading video: {str(e)}")
        raise


def preprocess_frame(
    frame: NDArray,
    equalize_hist: bool = True,
    blur_kernel_size: int = 5
) -> Optional[NDArray]:
    """Preprocess frame for edge detection.

    Args:
        frame: Input frame to process
        equalize_hist: Whether to apply histogram equalization
        blur_kernel_size: Size of Gaussian blur kernel

    Returns:
        Preprocessed grayscale frame

    Raises:
        ValueError: If input frame is invalid or empty
    """
    try:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame")

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_frame, (blur_kernel_size, blur_kernel_size), 0)
        
        # Apply histogram equalization if requested
        if equalize_hist:
            return cv2.equalizeHist(blurred)
        return blurred

    except Exception as e:
        logger.error(f"Error preprocessing frame: {str(e)}")
        return None


def detect_edges(
    frame: NDArray,
    config: VideoProcessingConfig
) -> Optional[NDArray]:
    """Detect edges using Canny edge detection.

    Args:
        frame: Input grayscale frame
        config: Processing configuration parameters

    Returns:
        Binary edge map

    Raises:
        ValueError: If input frame is invalid
    """
    try:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame")

        return cv2.Canny(
            frame,
            config.edge_low_threshold,
            config.edge_high_threshold
        )
    except Exception as e:
        logger.error(f"Error detecting edges: {str(e)}")
        return None


def find_and_analyze_contours(
    edges: NDArray,
    original_frame: NDArray,
    config: VideoProcessingConfig
) -> Optional[ContourAnalysisResult]:
    """Find contours and calculate Hu moments.

    Args:
        edges: Binary edge map
        original_frame: Original frame for drawing
        config: Processing configuration parameters

    Returns:
        ContourAnalysisResult object containing analysis results

    Raises:
        ValueError: If input images are invalid
    """
    try:
        if edges is None or edges.size == 0:
            raise ValueError("Invalid edge map")

        # Find contours
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Create empty image for contours
        contour_img = np.zeros_like(original_frame)
        hu_moments_list = []
        total_area = 0.0
        valid_contours = []

        # Process each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= config.min_contour_area:
                valid_contours.append(contour)
                total_area += area
                moments = cv2.moments(contour)
                hu_moments = cv2.HuMoments(moments)
                hu_moments_list.extend(hu_moments.flatten())

        # Draw valid contours
        cv2.drawContours(contour_img, valid_contours, -1, (0, 255, 0), 2)

        return ContourAnalysisResult(
            contour_image=contour_img,
            hu_moments=hu_moments_list,
            contour_count=len(valid_contours),
            total_area=total_area
        )

    except Exception as e:
        logger.error(f"Error analyzing contours: {str(e)}")
        return None


def create_visualization(
    original: NDArray,
    contour_result: ContourAnalysisResult,
    config: VideoProcessingConfig
) -> Optional[NDArray]:
    """Create side-by-side visualization of original and contour images.

    Args:
        original: Original frame
        contour_result: Results from contour analysis
        config: Processing configuration parameters

    Returns:
        Combined and resized visualization

    Raises:
        ValueError: If input images are invalid
    """
    try:
        if original is None or contour_result.contour_image is None:
            raise ValueError("Invalid input images")

        # Add text with contour information
        info_img = original.copy()
        cv2.putText(
            info_img,
            f"Contours: {contour_result.contour_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.putText(
            info_img,
            f"Total Area: {contour_result.total_area:.1f}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Combine and resize
        combined = cv2.hconcat([info_img, contour_result.contour_image])
        if config.display_scale != 1.0:
            new_size = (
                int(combined.shape[1] * config.display_scale),
                int(combined.shape[0] * config.display_scale)
            )
            combined = cv2.resize(combined, new_size)

        return combined

    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return None


def process_video_stream(
    cap: cv2.VideoCapture,
    config: VideoProcessingConfig
) -> bool:
    """Process video stream with edge detection and contour analysis.

    Args:
        cap: OpenCV VideoCapture object
        config: Processing configuration parameters

    Returns:
        True if processing completed successfully, False otherwise

    Raises:
        RuntimeError: If video stream cannot be processed
    """
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            logger.debug(f"Processing frame {frame_count}")

            # Preprocess frame
            processed_frame = preprocess_frame(frame)
            if processed_frame is None:
                continue

            # Detect edges
            edges = detect_edges(processed_frame, config)
            if edges is None:
                continue

            # Analyze contours
            contour_result = find_and_analyze_contours(
                edges,
                frame,
                config
            )
            if contour_result is None:
                continue

            # Create visualization
            visualization = create_visualization(
                frame,
                contour_result,
                config
            )
            if visualization is None:
                continue

            # Display results
            cv2.imshow('Edge Detection and Contour Analysis', visualization)

            # Check for exit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        return True

    except Exception as e:
        logger.error(f"Error processing video stream: {str(e)}")
        return False

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main(video_path: str) -> int:
    """Main function to run edge detection and contour analysis.

    Args:
        video_path: Path to the video file

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Initialize configuration
        config = VideoProcessingConfig(
            edge_low_threshold=50,
            edge_high_threshold=150,
            min_contour_area=100.0,
            visualization_size=(1400, 700),
            display_scale=0.8
        )

        # Load video
        cap, properties = load_video(video_path)
        logger.info(
            f"Processing video: {video_path}\n"
            f"FPS: {properties['fps']:.2f}\n"
            f"Frame count: {properties['frame_count']}\n"
            f"Duration: {properties['duration']:.2f}s"
        )

        # Process video stream
        if process_video_stream(cap, config):
            logger.info("Video processing completed successfully")
            return 0
        else:
            logger.error("Video processing failed")
            return 1

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python 08_edge_detection.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    sys.exit(main(video_path))
