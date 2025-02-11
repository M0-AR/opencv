"""Feature Extraction Module with Free Algorithms.

This module implements feature detection and description using only freely
available algorithms (non-patented) for video frame analysis. It includes:
- FAST (Features from Accelerated Segment Test)
- BRIEF (Binary Robust Independent Elementary Features)
- ORB (Oriented FAST and Rotated BRIEF)
- BRISK (Binary Robust Invariant Scalable Keypoints)

The module processes video frames and saves visualizations of detected features
for comparison and analysis.

Date: 2025-02-11
"""

from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import logging
import os
import sys
from pathlib import Path

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
class DetectorConfig:
    """Configuration for feature detectors."""
    max_features: int = 500
    scale_factor: float = 1.2
    n_levels: int = 8


@dataclass
class ProcessingConfig:
    """Configuration parameters for video processing."""
    crop_start: int = 200
    output_dir: str = "feature_extraction_output"
    save_format: str = "png"
    detector_colors: Dict[str, Tuple[int, int, int]] = None
    detector_params: Dict[str, DetectorConfig] = None

    def __post_init__(self):
        """Initialize default values that can't be set as default parameters."""
        if self.detector_colors is None:
            self.detector_colors = {
                'fast_brief': (0, 0, 255),  # Red
                'fast': (0, 0, 255),        # Red (fallback)
                'orb': (255, 255, 0),       # Cyan
                'brisk': (0, 255, 255)      # Yellow
            }
        if self.detector_params is None:
            self.detector_params = {
                'orb': DetectorConfig(max_features=1000),
                'brisk': DetectorConfig(n_levels=4),
                'fast': DetectorConfig()
            }


class FeatureDetectionResult(NamedTuple):
    """Results from feature detection."""
    detector_name: str
    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[NDArray]
    visualization: NDArray
    processing_time: float


class FreeFeatureDetector:
    """Class to handle feature detection using free algorithms."""

    def __init__(self, config: ProcessingConfig):
        """Initialize feature detectors and configuration.

        Args:
            config: Configuration parameters

        Raises:
            RuntimeError: If no detectors can be initialized
        """
        self.config = config
        self.detectors = {}
        self._initialize_detectors()

    def _initialize_detectors(self) -> None:
        """Initialize available feature detectors.

        Raises:
            RuntimeError: If no detectors can be initialized
        """
        try:
            # Initialize FAST and BRIEF
            self.detectors['fast'] = cv2.FastFeatureDetector_create()
            if hasattr(cv2.xfeatures2d, 'BriefDescriptorExtractor_create'):
                self.detectors['brief'] = cv2.xfeatures2d.BriefDescriptorExtractor_create()
                logger.info("BRIEF descriptor available")
            else:
                logger.warning("BRIEF descriptor not available")

            # Initialize ORB with custom parameters
            orb_config = self.config.detector_params['orb']
            self.detectors['orb'] = cv2.ORB_create(
                nfeatures=orb_config.max_features,
                scaleFactor=orb_config.scale_factor,
                nlevels=orb_config.n_levels
            )

            # Initialize BRISK with custom parameters
            brisk_config = self.config.detector_params['brisk']
            self.detectors['brisk'] = cv2.BRISK_create(
                thresh=10,
                octaves=brisk_config.n_levels
            )

            if not self.detectors:
                raise RuntimeError("No feature detectors could be initialized")

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
            FeatureDetectionResult containing detection results

        Raises:
            ValueError: If detector name is invalid or frame is empty
        """
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Invalid input frame")

            start_time = cv2.getTickCount()

            if detector_name == 'fast_brief':
                if 'brief' not in self.detectors:
                    return None
                keypoints = self.detectors['fast'].detect(frame, None)
                keypoints, descriptors = self.detectors['brief'].compute(frame, keypoints)
            elif detector_name in self.detectors:
                keypoints, descriptors = self.detectors[detector_name].detectAndCompute(frame, None)
            else:
                raise ValueError(f"Unknown detector: {detector_name}")

            # Calculate processing time
            processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()

            # Create visualization
            color = self.config.detector_colors.get(detector_name, (255, 255, 255))
            visualization = cv2.drawKeypoints(
                frame,
                keypoints,
                None,
                color=color,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

            # Add text with keypoint count and processing time
            cv2.putText(
                visualization,
                f"Keypoints: {len(keypoints)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )
            cv2.putText(
                visualization,
                f"Time: {processing_time*1000:.1f}ms",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

            return FeatureDetectionResult(
                detector_name=detector_name,
                keypoints=keypoints,
                descriptors=descriptors,
                visualization=visualization,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Error processing frame with {detector_name}: {str(e)}")
            return None

    def save_result(
        self,
        result: FeatureDetectionResult,
        frame_number: int
    ) -> bool:
        """Save feature detection results.

        Args:
            result: Feature detection results
            frame_number: Current frame number

        Returns:
            True if save successful, False otherwise
        """
        try:
            filename = os.path.join(
                self.config.output_dir,
                f"frame_{result.detector_name}_{frame_number:04d}.{self.config.save_format}"
            )
            cv2.imwrite(filename, result.visualization)
            return True
        except Exception as e:
            logger.error(f"Error saving result for {result.detector_name}: {str(e)}")
            return False


def process_video(
    video_path: str,
    config: ProcessingConfig
) -> bool:
    """Process video file with multiple feature detectors.

    Args:
        video_path: Path to the input video file
        config: Processing configuration

    Returns:
        True if processing successful, False otherwise

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If video cannot be processed
    """
    try:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Initialize video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Processing video: {video_path.name} ({fps:.1f} fps, {total_frames} frames)")

        # Initialize detector
        detector = FreeFeatureDetector(config)
        frame_count = 0
        processing_stats = {name: [] for name in config.detector_colors.keys()}

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
                if result:
                    processing_stats[detector_name].append(result.processing_time)
                    if not detector.save_result(result, frame_count):
                        logger.warning(f"Failed to save result for {detector_name}")

            # Log progress
            frame_count += 1
            if frame_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

        # Log statistics
        for name, times in processing_stats.items():
            if times:
                avg_time = np.mean(times) * 1000  # Convert to ms
                logger.info(f"{name}: Average processing time = {avg_time:.1f}ms")

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
        config = ProcessingConfig(
            crop_start=200,
            output_dir="07_02",
            save_format="png"
        )

        # Process video
        if len(sys.argv) != 2:
            logger.error("Usage: python 07_02_feature_extraction.py <video_path>")
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
