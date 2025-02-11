"""Feature Matching Module.

This module implements robust feature matching for video frame analysis,
particularly useful for detecting duplicate or similar frames in video streams.
It uses ORB (Oriented FAST and Rotated BRIEF) features for efficient
matching and comparison.

The module handles real-world scenarios where frames might have:
- Slight movements
- Lighting changes
- Camera noise
- Perspective variations

Author: OpenCV Toolkit Team
Date: 2025-02-11
"""

from typing import List, Tuple, Optional
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


class FeatureMatcher:
    """Handles feature detection and matching for video frames."""

    def __init__(
        self,
        match_ratio: float = 0.75,
        min_matches: int = 10
    ):
        """Initialize the feature matcher.

        Args:
            match_ratio: Ratio test threshold for feature matching (0.0-1.0)
            min_matches: Minimum number of good matches to consider frames similar
        """
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher()
        self.seen_frames_features: List[Tuple[
            List[cv2.KeyPoint],
            NDArray
        ]] = []
        self.match_ratio = match_ratio
        self.min_matches = min_matches

    def detect_features(
        self,
        frame: NDArray
    ) -> Tuple[Optional[List[cv2.KeyPoint]], Optional[NDArray]]:
        """Detect ORB features in a frame.

        Args:
            frame: Input grayscale frame

        Returns:
            Tuple of keypoints and descriptors, or (None, None) if detection fails
        """
        try:
            return self.orb.detectAndCompute(frame, None)
        except Exception as e:
            logger.error(f"Error detecting features: {str(e)}")
            return None, None

    def is_frame_similar(
        self,
        desc1: NDArray,
        desc2: NDArray
    ) -> bool:
        """Check if two frames are similar based on their descriptors.

        Args:
            desc1: Descriptors of first frame
            desc2: Descriptors of second frame

        Returns:
            True if frames are similar, False otherwise
        """
        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good_matches = [
                m for m, n in matches
                if m.distance < self.match_ratio * n.distance
            ]
            return len(good_matches) >= self.min_matches
        except Exception as e:
            logger.error(f"Error matching features: {str(e)}")
            return False

    def is_frame_seen(
        self,
        frame: NDArray
    ) -> bool:
        """Check if a frame has been seen before.

        Args:
            frame: Input grayscale frame to check

        Returns:
            True if frame is similar to a previously seen frame
        """
        keypoints_new, descriptors_new = self.detect_features(frame)
        if keypoints_new is None or descriptors_new is None:
            return False

        for _, desc in self.seen_frames_features:
            if self.is_frame_similar(desc, descriptors_new):
                return True

        self.seen_frames_features.append((keypoints_new, descriptors_new))
        return False


def setup_output_directory(path: str) -> bool:
    """Create output directory if it doesn't exist.

    Args:
        path: Directory path to create

    Returns:
        True if directory exists or was created successfully
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {path}: {str(e)}")
        return False


def load_video(video_path: str) -> Tuple[Optional[cv2.VideoCapture], bool]:
    """Load and validate a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Tuple containing:
            - VideoCapture object if successful, None otherwise
            - Boolean indicating success status
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")
        return cap, True
    except Exception as e:
        logger.error(f"Error loading video: {str(e)}")
        return None, False


def process_video(
    cap: cv2.VideoCapture,
    output_dir: str,
    crop_start: int = 200
) -> None:
    """Process video frames and save unique frames.

    Args:
        cap: OpenCV VideoCapture object
        output_dir: Directory to save unique frames
        crop_start: Starting point for horizontal cropping
    """
    matcher = FeatureMatcher()
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Crop and convert frame
            cropped_frame = frame[:, crop_start:]
            gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

            frame_count += 1

            # Check for duplicate frames
            if matcher.is_frame_seen(gray_frame):
                logger.debug(f"Frame {frame_count} is duplicate, skipping...")
                continue

            logger.info(f"Processing new frame {frame_count}")
            
            # Save unique frame
            frame_path = os.path.join(
                output_dir,
                f"frame_{frame_count:05d}.jpg"
            )
            cv2.imwrite(frame_path, cropped_frame)

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
    finally:
        cap.release()


def main(video_path: str, output_dir: str) -> int:
    """Main function to run feature matching on video.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save unique frames

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Setup output directory
        if not setup_output_directory(output_dir):
            return 1

        # Load video
        cap, success = load_video(video_path)
        if not success:
            return 1

        # Process video
        process_video(cap, output_dir)
        return 0

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    VIDEO_PATH = '020146-3173 (35).mp4'
    OUTPUT_DIR = '05_feature_matching'
    sys.exit(main(VIDEO_PATH, OUTPUT_DIR))
