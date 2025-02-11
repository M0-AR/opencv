"""Template Matching Module.

This module demonstrates basic frame analysis using frame differencing to identify
unique frames in a video stream. It uses a simple but effective approach to detect
significant changes between consecutive frames.

Algorithm Explanation for Beginners:
--------------------------------
This program uses a straightforward technique to find unique frames:

1. Frame Preparation:
   - We convert each frame to grayscale (black and white)
   - We remove any black borders or unwanted areas
   - We can apply histogram equalization to improve contrast
   
2. Frame Differencing:
   - We compare each new frame with the previous frame
   - We calculate the absolute difference between frames
   - This shows us where changes have occurred
   
3. Change Detection:
   - We calculate the average difference between frames
   - If this difference is above a threshold, we consider it a unique frame
   - Higher threshold = less sensitive to changes
   - Lower threshold = more sensitive to changes

The result shows:
- Left: Previous frame
- Right: Current frame
- Saved frames are stored in the 'unique_frames' directory

This helps us identify frames where significant changes occur in the video.

Key Features:
- Real-time frame comparison
- Adjustable detection sensitivity
- Histogram equalization option
- Frame saving capability
- Progress tracking

Usage:
    python 08_Template_Matching_uniqe.py <video_path>

Controls:
    'q' - quit
    's' - save current frame
    '+' - increase detection sensitivity
    '-' - decrease detection sensitivity
    'r' - reset parameters to default
    'h' - toggle histogram equalization
"""

import logging
import os
from pathlib import Path
import sys
from typing import Optional, Tuple
from dataclasses import dataclass

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
class DetectionParams:
    """Parameters for frame detection."""
    difference_threshold: float = 40.0  # Mean difference threshold
    min_threshold: float = 10.0
    max_threshold: float = 100.0
    use_histogram: bool = True
    
    def adjust_sensitivity(self, increase: bool = True) -> None:
        """Adjust detection sensitivity.
        
        Args:
            increase: If True, increase sensitivity; if False, decrease
        """
        factor = 0.8 if increase else 1.2
        new_threshold = self.difference_threshold * factor
        
        if self.min_threshold <= new_threshold <= self.max_threshold:
            self.difference_threshold = new_threshold
            
    def reset(self) -> None:
        """Reset parameters to default values."""
        self.__init__()


class FrameChangeDetector:
    """A class to handle frame change detection and visualization."""

    def __init__(self, output_dir: str = "unique_frames"):
        """Initialize the detector.

        Args:
            output_dir: Directory to save unique frames
        """
        self.output_dir = Path(output_dir)
        self.frame_count = 0
        self.unique_count = 0
        self.window_name = 'Current and Previous Frames'
        self.params = DetectionParams()
        self.prev_frame: Optional[NDArray] = None
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

    def preprocess_frame(self, frame: NDArray) -> NDArray:
        """Preprocess frame before analysis.

        Args:
            frame: Input frame

        Returns:
            Preprocessed grayscale frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Remove black area if present
        gray = gray[:, 200:]
        
        # Apply histogram equalization if enabled
        if self.params.use_histogram:
            gray = cv2.equalizeHist(gray)
            
        return gray

    def is_frame_unique(self, frame: NDArray) -> bool:
        """Check if frame is unique using frame differencing.

        Args:
            frame: Current frame

        Returns:
            True if frame is unique, False otherwise
        """
        if self.prev_frame is None:
            return True
            
        # Calculate absolute difference
        frame_diff = cv2.absdiff(self.prev_frame, frame)
        mean_diff = np.mean(frame_diff)
        
        return mean_diff > self.params.difference_threshold

    def create_display_frame(
        self,
        current: NDArray,
        previous: Optional[NDArray] = None,
        target_width: int = 1400,
        target_height: int = 700
    ) -> NDArray:
        """Create display frame with both views.

        Args:
            current: Current frame
            previous: Previous frame
            target_width: Desired width
            target_height: Desired height

        Returns:
            Combined display frame
        """
        if previous is None:
            previous = np.zeros_like(current)
            
        # Convert to BGR for display
        current_bgr = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)
        previous_bgr = cv2.cvtColor(previous, cv2.COLOR_GRAY2BGR)
        
        # Combine side by side
        combined = cv2.hconcat([previous_bgr, current_bgr])
        
        # Resize to target dimensions
        return cv2.resize(combined, (target_width, target_height))

    def save_frame(self, frame: NDArray) -> None:
        """Save unique frame.

        Args:
            frame: Frame to save
        """
        filename = self.output_dir / f"unique_frame_{self.unique_count:04d}.jpg"
        cv2.imwrite(str(filename), frame)
        logger.info(f"Saved unique frame to: {filename}")
        self.unique_count += 1

    def process_video(self, video_path: str) -> bool:
        """Process video and detect unique frames.

        Args:
            video_path: Path to input video

        Returns:
            True if successful, False otherwise
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error("Could not open video file")
                return False

            logger.info("Starting video processing...")
            logger.info("Controls:")
            logger.info("  'q' - quit")
            logger.info("  's' - save current frame")
            logger.info("  '+' - increase detection sensitivity")
            logger.info("  '-' - decrease detection sensitivity")
            logger.info("  'r' - reset parameters")
            logger.info("  'h' - toggle histogram equalization")

            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                current_frame = self.preprocess_frame(frame)
                
                # Check if frame is unique
                if self.is_frame_unique(current_frame):
                    self.save_frame(frame)
                
                # Create display frame
                display_frame = self.create_display_frame(
                    current_frame,
                    self.prev_frame
                )

                # Add parameter info
                hist = "On" if self.params.use_histogram else "Off"
                cv2.putText(
                    display_frame,
                    f"Thresh: {self.params.difference_threshold:.1f} | "
                    f"Unique: {self.unique_count} | "
                    f"Hist: {hist}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                # Show result
                cv2.imshow(self.window_name, display_frame)

                # Update previous frame
                self.prev_frame = current_frame.copy()

                # Handle keyboard input
                key = cv2.waitKey(25) & 0xFF
                if key == ord('q'):
                    logger.info("Quitting...")
                    break
                elif key == ord('s'):
                    self.save_frame(frame)
                elif key == ord('+'):
                    self.params.adjust_sensitivity(increase=True)
                    logger.info(f"Increased sensitivity: {self.params.difference_threshold:.1f}")
                elif key == ord('-'):
                    self.params.adjust_sensitivity(increase=False)
                    logger.info(f"Decreased sensitivity: {self.params.difference_threshold:.1f}")
                elif key == ord('r'):
                    self.params.reset()
                    logger.info("Reset parameters to default")
                elif key == ord('h'):
                    self.params.use_histogram = not self.params.use_histogram
                    hist = "enabled" if self.params.use_histogram else "disabled"
                    logger.info(f"Histogram equalization {hist}")

                self.frame_count += 1

            logger.info(f"Total frames processed: {self.frame_count}")
            logger.info(f"Unique frames saved: {self.unique_count}")
            return True

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return False

        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()


def main() -> int:
    """Main function.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Check command line arguments
        if len(sys.argv) != 2:
            logger.error("Usage: python 08_Template_Matching_uniqe.py <video_path>")
            return 1

        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return 1

        # Create detector and process video
        detector = FrameChangeDetector()
        if detector.process_video(video_path):
            logger.info("Processing completed successfully")
            return 0
        else:
            logger.error("Processing failed")
            return 1

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
