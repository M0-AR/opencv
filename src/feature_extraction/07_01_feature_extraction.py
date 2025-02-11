"""Feature Extraction and Comparison Module.

This module implements various feature detection and description algorithms
for video frame analysis. It supports multiple feature detectors including:
- SIFT (Scale-Invariant Feature Transform)
- FAST (Features from Accelerated Segment Test) with BRIEF descriptors
- ORB (Oriented FAST and Rotated BRIEF)
- BRISK (Binary Robust Invariant Scalable Keypoints)

The module processes video frames and saves visualizations of detected features
using different algorithms for comparison.

Date: 2025-02-11
"""

from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import logging
import os
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


@dataclass
class FeatureDetectionConfig:
    """Configuration parameters for feature detection."""
    crop_start: int = 200  # Start column for frame cropping
    output_dir: str = "feature_extraction_output"
    save_format: str = "png"
    detector_colors: Dict[str, Tuple[int, int, int]] = None

    def __post_init__(self):
        """Initialize default values that can't be set as default parameters."""
        if self.detector_colors is None:
            self.detector_colors = {
                'sift': (255, 0, 0),      # Blue
                'fast_brief': (0, 0, 255), # Red
                'orb': (255, 255, 0),      # Cyan
                'brisk': (0, 255, 255)     # Yellow
            }


class FeatureDetectionResult(NamedTuple):
    """Results from feature detection."""
    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[NDArray]
    visualization: NDArray


class FeatureDetector:
    """Class to handle feature detection and visualization."""

    def __init__(self, config: FeatureDetectionConfig):
        """Initialize feature detectors and configuration.

        Args:
            config: Configuration parameters for feature detection
        """
        self.config = config
        self._initialize_detectors()

    def _initialize_detectors(self) -> None:
        """Initialize all feature detectors and descriptors."""
        try:
            self.detectors = {
                'sift': cv2.SIFT_create(),
                'fast': cv2.FastFeatureDetector_create(),
                'brief': cv2.xfeatures2d.BriefDescriptorExtractor_create(),
                'orb': cv2.ORB_create(),
                'brisk': cv2.BRISK_create()
            }
            logger.info("Feature detectors initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing detectors: {str(e)}")
            raise

    def process_frame(
        self,
        frame: NDArray,
        detector_name: str
    ) -> Optional[FeatureDetectionResult]:
        """Process a frame with specified feature detector.

        Args:
            frame: Input frame to process
            detector_name: Name of the detector to use

        Returns:
            FeatureDetectionResult containing keypoints, descriptors, and visualization

        Raises:
            ValueError: If detector name is invalid or frame is empty
        """
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Invalid input frame")

            if detector_name not in self.config.detector_colors:
                raise ValueError(f"Unknown detector: {detector_name}")

            # Detect and compute features
            if detector_name == 'fast_brief':
                keypoints = self.detectors['fast'].detect(frame, None)
                keypoints, descriptors = self.detectors['brief'].compute(frame, keypoints)
            else:
                keypoints, descriptors = self.detectors[detector_name].detectAndCompute(frame, None)

            # Create visualization
            color = self.config.detector_colors[detector_name]
            visualization = cv2.drawKeypoints(frame, keypoints, None, color=color)

            return FeatureDetectionResult(keypoints, descriptors, visualization)

        except Exception as e:
            logger.error(f"Error processing frame with {detector_name}: {str(e)}")
            return None

    def save_visualization(
        self,
        result: FeatureDetectionResult,
        detector_name: str,
        frame_number: int
    ) -> bool:
        """Save feature detection visualization.

        Args:
            result: Feature detection results
            detector_name: Name of the detector used
            frame_number: Current frame number

        Returns:
            True if save successful, False otherwise
        """
        try:
            filename = os.path.join(
                self.config.output_dir,
                f"frame_{detector_name}_{frame_number:04d}.{self.config.save_format}"
            )
            cv2.imwrite(filename, result.visualization)
            return True
        except Exception as e:
            logger.error(f"Error saving visualization for {detector_name}: {str(e)}")
            return False


def process_video(
    video_path: str,
    config: FeatureDetectionConfig
) -> bool:
    """Process video file with multiple feature detectors.

    Args:
        video_path: Path to the input video file
        config: Configuration parameters

    Returns:
        True if processing successful, False otherwise

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If video cannot be processed
    """
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        # Initialize feature detector
        detector = FeatureDetector(config)
        frame_count = 0

        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Crop frame if needed
            if config.crop_start > 0:
                frame = frame[:, config.crop_start:]

            # Process frame with each detector
            for detector_name in config.detector_colors.keys():
                result = detector.process_frame(frame, detector_name)
                if result and not detector.save_visualization(result, detector_name, frame_count):
                    logger.warning(f"Failed to save visualization for {detector_name}")

            frame_count += 1
            if frame_count % 10 == 0:
                logger.info(f"Processed {frame_count} frames")

        logger.info(f"Total frames processed: {frame_count}")
        return True

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return False

    finally:
        if 'cap' in locals():
            cap.release()


def main() -> int:
    """Main function to run feature extraction.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Initialize configuration
        config = FeatureDetectionConfig(
            crop_start=200,
            output_dir="07",
            save_format="png"
        )

        # Process video
        if len(sys.argv) != 2:
            logger.error("Usage: python 07_01_feature_extraction.py <video_path>")
            return 1

        video_path = sys.argv[1]
        if process_video(video_path, config):
            logger.info(f"Feature extraction completed. Results saved to {config.output_dir}")
            return 0
        else:
            logger.error("Feature extraction failed")
            return 1

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())