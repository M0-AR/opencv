"""Hough Circle Transform Detection Module.

A simple and educational module that demonstrates circle detection in video frames
using the Hough Circle Transform algorithm. It shows the original video with
detected circles overlaid in real-time.

Key Features:
- Real-time circle detection
- Adjustable detection parameters
- Interactive parameter tuning
- Circle visualization with configurable colors
- Simple keyboard controls for interaction

Usage:
    python 08_hough_circle_transform.py <video_path>

Controls:
    'q' - quit
    's' - save current frame
    '+' - increase detection sensitivity
    '-' - decrease detection sensitivity
    'r' - reset parameters to default
    'c' - cycle through color schemes

Date: 2025-02-11
"""

import logging
import os
from pathlib import Path
import sys
from typing import List, Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


# Set up logging with a simple format
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format for better readability
)
logger = logging.getLogger(__name__)


@dataclass
class CircleDetectionParams:
    """Parameters for Hough Circle detection."""
    dp: float = 1.2  # Inverse ratio of accumulator resolution
    min_dist: int = 100  # Minimum distance between centers
    param1: int = 50  # Upper threshold for edge detection
    param2: int = 30  # Threshold for center detection
    min_radius: int = 0  # Minimum radius to detect
    max_radius: int = 0  # Maximum radius to detect (0 = no limit)
    
    def adjust_sensitivity(self, increase: bool = True) -> None:
        """Adjust detection sensitivity.
        
        Args:
            increase: If True, increase sensitivity; if False, decrease
        """
        factor = 1.1 if increase else 0.9
        self.param1 = int(self.param1 * factor)
        self.param2 = int(self.param2 * factor)
        
    def reset(self) -> None:
        """Reset parameters to default values."""
        self.__init__()


class CircleDetector:
    """A simple class to handle circle detection visualization."""

    def __init__(self, output_dir: str = "circle_detection_output"):
        """Initialize the detector.

        Args:
            output_dir: Directory to save output images (if requested)
        """
        self.output_dir = Path(output_dir)
        self.frame_count = 0
        self.window_name = 'Hough Circle Detection'
        self.params = CircleDetectionParams()
        self.color_scheme_idx = 0
        self.color_schemes = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (0, 255, 255),  # Yellow
        ]
        
        # Create output directory if saving is needed
        self.output_dir.mkdir(exist_ok=True)

    def detect_circles(self, frame: NDArray) -> Optional[NDArray]:
        """Detect circles in a frame using Hough Circle Transform.

        Args:
            frame: Input frame

        Returns:
            Array of detected circles (x, y, radius) or None if none found
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.params.dp,
            minDist=self.params.min_dist,
            param1=self.params.param1,
            param2=self.params.param2,
            minRadius=self.params.min_radius,
            maxRadius=self.params.max_radius
        )
        
        if circles is not None:
            return np.round(circles[0, :]).astype("int")
        return None

    def draw_circles(self, frame: NDArray, circles: Optional[NDArray]) -> NDArray:
        """Draw detected circles on the frame.

        Args:
            frame: Input frame
            circles: Array of circles to draw (x, y, radius)

        Returns:
            Frame with circles drawn
        """
        result = frame.copy()
        
        if circles is not None:
            color = self.color_schemes[self.color_scheme_idx]
            
            # Draw each circle
            for (x, y, r) in circles:
                # Draw the circle outline
                cv2.circle(result, (x, y), r, color, 2)
                # Draw the circle center
                cv2.circle(result, (x, y), 2, color, 3)
        
        return result

    def save_frame(self, frame: NDArray) -> None:
        """Save the current frame.

        Args:
            frame: Frame to save
        """
        filename = self.output_dir / f"frame_{self.frame_count:04d}.png"
        cv2.imwrite(str(filename), frame)
        logger.info(f"Saved frame to: {filename}")

    def process_video(self, video_path: str) -> bool:
        """Process video and show circle detection.

        Args:
            video_path: Path to input video file

        Returns:
            True if processing was successful, False otherwise
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
            logger.info("  'c' - cycle through colors")

            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Remove black area if present
                frame = frame[:, 200:]

                # Detect and draw circles
                circles = self.detect_circles(frame)
                display_frame = self.draw_circles(frame, circles)

                # Add parameter info
                cv2.putText(
                    display_frame,
                    f"Sensitivity: {self.params.param1}, {self.params.param2}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    self.color_schemes[self.color_scheme_idx],
                    2
                )

                # Resize for display
                display_frame = cv2.resize(display_frame, (1400, 700))

                # Show result
                cv2.imshow(self.window_name, display_frame)

                # Handle keyboard input
                key = cv2.waitKey(25) & 0xFF
                if key == ord('q'):
                    logger.info("Quitting...")
                    break
                elif key == ord('s'):
                    self.save_frame(display_frame)
                elif key == ord('+'):
                    self.params.adjust_sensitivity(increase=True)
                    logger.info(f"Increased sensitivity: {self.params.param1}, {self.params.param2}")
                elif key == ord('-'):
                    self.params.adjust_sensitivity(increase=False)
                    logger.info(f"Decreased sensitivity: {self.params.param1}, {self.params.param2}")
                elif key == ord('r'):
                    self.params.reset()
                    logger.info("Reset parameters to default")
                elif key == ord('c'):
                    self.color_scheme_idx = (self.color_scheme_idx + 1) % len(self.color_schemes)
                    logger.info("Changed circle color")

                self.frame_count += 1

            return True

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return False

        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()


def main() -> int:
    """Main function to run circle detection visualization.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Check command line arguments
        if len(sys.argv) != 2:
            logger.error("Usage: python 08_hough_circle_transform.py <video_path>")
            return 1

        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return 1

        # Create detector and process video
        detector = CircleDetector()
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
