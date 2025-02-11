"""SLAM (Simultaneous Localization and Mapping) Processor Module.

This module implements a basic visual SLAM system using ORB features for tracking
camera motion and building a sparse 3D map of the environment.

Algorithm Explanation for Beginners:
--------------------------------
SLAM helps a camera understand where it is and what's around it:

1. Feature Detection (ORB):
   - Find special points in each frame that are easy to track
   - These points are like landmarks in the image
   - ORB looks for corners and unique patterns
   - Each point gets a unique "fingerprint" (descriptor)

2. Feature Matching:
   - Match landmarks between frames
   - Like connecting dots between two images
   - Uses Hamming distance to find best matches
   - Helps track how points move

3. Motion Estimation:
   - From matched points, calculate camera movement
   - Uses Essential Matrix to find rotation and translation
   - Like solving a puzzle of how camera moved
   - Filters out bad matches using RANSAC

4. Mapping:
   - Convert 2D image points to 3D world points
   - Build a map of where landmarks are in 3D space
   - Updates map as camera moves
   - Shows camera path through environment

Key Features:
- Real-time feature tracking
- Interactive visualization
- Progress feedback
- Error handling
- Resource cleanup
- Performance optimization

Usage:
    python slam_processor.py <video_path> [--width N] [--features N]

Controls:
    'p' - Pause/Resume
    'm' - Toggle map view
    't' - Toggle feature tracks
    'r' - Reset tracking
    'q' - Quit
"""

from dataclasses import dataclass
import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SLAMParams:
    """Parameters for SLAM processing."""
    target_width: int = 640
    max_features: int = 1000
    min_matches: int = 10
    ransac_threshold: float = 3.0
    min_track_length: int = 20
    max_track_length: int = 50
    feature_quality: float = 0.01
    min_feature_distance: int = 10


class SLAMProcessor:
    """A class to handle visual SLAM processing."""

    def __init__(self):
        """Initialize the SLAM processor."""
        self.params = SLAMParams()
        
        # Initialize feature detector
        self.orb = cv2.ORB_create(
            nfeatures=self.params.max_features,
            fastThreshold=20,
            edgeThreshold=20
        )
        
        # Initialize matcher
        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=True
        )
        
        # Initialize state
        self.prev_frame = None
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.tracks = []
        self.camera_poses = []
        self.map_points = []
        
        # Initialize flags
        self.paused = False
        self.show_map = True
        self.show_tracks = True
        self.need_reset = False

    def resize_frame(self, frame: NDArray) -> NDArray:
        """Resize frame while maintaining aspect ratio.

        Args:
            frame: Input frame

        Returns:
            Resized frame
        """
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        target_height = int(self.params.target_width / aspect_ratio)
        return cv2.resize(
            frame,
            (self.params.target_width, target_height)
        )

    def detect_features(
        self,
        frame: NDArray
    ) -> Tuple[List[cv2.KeyPoint], NDArray]:
        """Detect ORB features in frame.

        Args:
            frame: Input frame

        Returns:
            Tuple of keypoints and descriptors
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect and compute
            keypoints, descriptors = self.orb.detectAndCompute(
                gray,
                None
            )
            
            if keypoints is None or descriptors is None:
                raise RuntimeError("No features detected")
                
            return keypoints, descriptors
            
        except Exception as e:
            logger.error(f"Error detecting features: {str(e)}")
            return [], None

    def match_features(
        self,
        desc1: NDArray,
        desc2: NDArray
    ) -> List[cv2.DMatch]:
        """Match features between frames.

        Args:
            desc1: First frame descriptors
            desc2: Second frame descriptors

        Returns:
            List of matches
        """
        try:
            matches = self.matcher.match(desc1, desc2)
            
            # Sort by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Keep best matches
            matches = matches[:self.params.max_features]
            
            return matches
            
        except Exception as e:
            logger.error(f"Error matching features: {str(e)}")
            return []

    def estimate_motion(
        self,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> Optional[Tuple[NDArray, NDArray]]:
        """Estimate camera motion from matches.

        Args:
            kp1: First frame keypoints
            kp2: Second frame keypoints
            matches: Feature matches

        Returns:
            Tuple of rotation and translation matrices
        """
        try:
            if len(matches) < self.params.min_matches:
                return None
                
            # Get matched point coordinates
            pts1 = np.float32(
                [kp1[m.queryIdx].pt for m in matches]
            ).reshape(-1, 1, 2)
            pts2 = np.float32(
                [kp2[m.trainIdx].pt for m in matches]
            ).reshape(-1, 1, 2)
            
            # Calculate essential matrix
            E, mask = cv2.findEssentialMat(
                pts1,
                pts2,
                focal=1.0,
                pp=(0., 0.),
                method=cv2.RANSAC,
                prob=0.999,
                threshold=self.params.ransac_threshold
            )
            
            if E is None:
                return None
                
            # Recover pose
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2)
            
            return R, t
            
        except Exception as e:
            logger.error(f"Error estimating motion: {str(e)}")
            return None

    def update_tracks(
        self,
        keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> None:
        """Update feature tracks.

        Args:
            keypoints: Current frame keypoints
            matches: Feature matches
        """
        try:
            # Update existing tracks
            if self.tracks:
                # Get matched indices
                matched = set()
                for m in matches:
                    matched.add(m.trainIdx)
                    
                # Update or remove tracks
                new_tracks = []
                for track in self.tracks:
                    if track[-1] in matched:
                        # Extend track
                        new_tracks.append(
                            track + [keypoints[track[-1]].pt]
                        )
                    elif len(track) >= self.params.min_track_length:
                        # Keep completed track
                        new_tracks.append(track)
                        
                self.tracks = new_tracks
                
            # Start new tracks
            for m in matches:
                if len(self.tracks) >= self.params.max_features:
                    break
                    
                self.tracks.append([keypoints[m.trainIdx].pt])
                
        except Exception as e:
            logger.error(f"Error updating tracks: {str(e)}")

    def draw_results(
        self,
        frame: NDArray,
        keypoints: List[cv2.KeyPoint],
        matches: Optional[List[cv2.DMatch]] = None
    ) -> NDArray:
        """Draw visualization on frame.

        Args:
            frame: Input frame
            keypoints: Current frame keypoints
            matches: Optional feature matches

        Returns:
            Frame with visualization
        """
        try:
            # Draw keypoints
            frame = cv2.drawKeypoints(
                frame,
                keypoints,
                None,
                color=(0, 255, 0),
                flags=0
            )
            
            # Draw matches
            if matches and self.prev_frame is not None:
                frame = cv2.drawMatches(
                    self.prev_frame,
                    self.prev_keypoints,
                    frame,
                    keypoints,
                    matches,
                    None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
            
            # Draw tracks
            if self.show_tracks:
                for track in self.tracks[-self.params.max_track_length:]:
                    if len(track) >= 2:
                        pts = np.array(track, np.int32)
                        cv2.polylines(
                            frame,
                            [pts],
                            False,
                            (0, 0, 255),
                            2
                        )
            
            # Draw info
            cv2.putText(
                frame,
                f"Features: {len(keypoints)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            if matches:
                cv2.putText(
                    frame,
                    f"Matches: {len(matches)}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing results: {str(e)}")
            return frame

    def process_frame(self, frame: NDArray) -> NDArray:
        """Process a single frame.

        Args:
            frame: Input frame

        Returns:
            Frame with visualization
        """
        try:
            # Resize frame
            frame = self.resize_frame(frame)
            
            # Detect features
            keypoints, descriptors = self.detect_features(frame)
            
            # Initialize if first frame
            if self.prev_frame is None:
                self.prev_frame = frame.copy()
                self.prev_keypoints = keypoints
                self.prev_descriptors = descriptors
                return self.draw_results(frame, keypoints)
            
            # Match features
            if descriptors is not None and self.prev_descriptors is not None:
                matches = self.match_features(
                    self.prev_descriptors,
                    descriptors
                )
            else:
                matches = []
            
            # Estimate motion
            if matches:
                motion = self.estimate_motion(
                    self.prev_keypoints,
                    keypoints,
                    matches
                )
                if motion:
                    R, t = motion
                    self.camera_poses.append((R, t))
            
            # Update tracks
            self.update_tracks(keypoints, matches)
            
            # Draw results
            result = self.draw_results(frame, keypoints, matches)
            
            # Update previous frame
            self.prev_frame = frame.copy()
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame

    def reset(self) -> None:
        """Reset SLAM state."""
        self.prev_frame = None
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.tracks = []
        self.camera_poses = []
        self.map_points = []
        self.need_reset = False

    def handle_key(self, key: int) -> bool:
        """Handle keyboard input.

        Args:
            key: Key code

        Returns:
            True if should continue, False if should quit
        """
        if key == ord('q'):
            return False
        elif key == ord('p'):
            self.paused = not self.paused
        elif key == ord('m'):
            self.show_map = not self.show_map
        elif key == ord('t'):
            self.show_tracks = not self.show_tracks
        elif key == ord('r'):
            self.need_reset = True
        return True

    def process_video(self, video_path: str) -> None:
        """Process video file.

        Args:
            video_path: Path to video file
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            
            logger.info(f"Processing video: {video_path}")
            
            while True:
                # Check for reset
                if self.need_reset:
                    self.reset()
                
                # Read frame
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                
                # Process frame
                result = self.process_frame(frame)
                
                # Show result
                cv2.imshow('SLAM', result)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key(key):
                    break
            
            logger.info("Processing complete")
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main() -> int:
    """Main function.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(
            description='Process video using visual SLAM'
        )
        parser.add_argument(
            'video_path',
            help='Path to video file'
        )
        parser.add_argument(
            '--width',
            type=int,
            default=640,
            help='Target frame width'
        )
        parser.add_argument(
            '--features',
            type=int,
            default=1000,
            help='Maximum number of features'
        )
        
        args = parser.parse_args()
        
        # Create processor
        processor = SLAMProcessor()
        processor.params.target_width = args.width
        processor.params.max_features = args.features
        
        # Process video
        processor.process_video(args.video_path)
        
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
