"""Hough Line Transform Detection Module.

A simple and educational module that demonstrates line detection in video frames
using the Hough Transform algorithm. It shows three views side by side:
1. Original grayscale image
2. Edge detection result
3. Detected lines overlaid on the original image

Key Features:
- Real-time line detection
- Adjustable detection parameters
- Edge detection visualization
- Interactive parameter tuning
- Line visualization with configurable colors
- Simple keyboard controls for interaction

Usage:
    python 08_hough_transform.py <video_path>

Controls:
    'q' - quit
    's' - save current frame
    '+' - increase detection sensitivity
    '-' - decrease detection sensitivity
    'r' - reset parameters to default
    'c' - cycle through color schemes
    'e' - toggle edge detection method (Canny/Sobel)

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
class LineDetectionParams:
    """Parameters for Hough Line detection."""
    rho: float = 1.0  # Distance resolution in pixels
    theta: float = np.pi / 180  # Angular resolution in radians
    threshold: int = 200  # Minimum number of votes
    min_line_length: int = 100  # Minimum line length
    max_line_gap: int = 10  # Maximum gap between line segments
    canny_low: int = 50  # Lower threshold for Canny
    canny_high: int = 150  # Upper threshold for Canny
    
    def adjust_sensitivity(self, increase: bool = True) -> None:
        """Adjust detection sensitivity.
        
        Args:
            increase: If True, increase sensitivity; if False, decrease
        """
        factor = 0.9 if increase else 1.1  # Inverse for threshold
        self.threshold = int(self.threshold * factor)
        self.canny_low = int(self.canny_low * factor)
        self.canny_high = int(self.canny_high * factor)
        
    def reset(self) -> None:
        """Reset parameters to default values."""
        self.__init__()


class LineDetector:
    """A simple class to handle line detection visualization."""

    def __init__(self, output_dir: str = "line_detection_output"):
        """Initialize the detector.

        Args:
            output_dir: Directory to save output images (if requested)
        """
        self.output_dir = Path(output_dir)
        self.frame_count = 0
        self.window_name = 'Original, Edges, and Hough Transform Lines'
        self.params = LineDetectionParams()
        self.color_scheme_idx = 0
        self.use_canny = True
        self.color_schemes = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (0, 255, 255),  # Yellow
        ]
        
        # Create output directory if saving is needed
        self.output_dir.mkdir(exist_ok=True)

    def detect_edges(self, frame: NDArray) -> NDArray:
        """Detect edges in the frame.

        Args:
            frame: Input grayscale frame

        Returns:
            Edge detection result
        """
        if self.use_canny:
            return cv2.Canny(
                frame,
                self.params.canny_low,
                self.params.canny_high,
                apertureSize=3
            )
        else:
            # Use Sobel edge detection
            sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            return cv2.normalize(
                magnitude,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U
            )

    def detect_lines(self, edges: NDArray) -> Optional[NDArray]:
        """Detect lines using Hough Transform.

        Args:
            edges: Edge detection result

        Returns:
            Array of detected lines or None if none found
        """
        return cv2.HoughLines(
            edges,
            self.params.rho,
            self.params.theta,
            self.params.threshold
        )

    def draw_lines(self, frame: NDArray, lines: Optional[NDArray]) -> NDArray:
        """Draw detected lines on the frame.

        Args:
            frame: Input frame
            lines: Array of detected lines

        Returns:
            Frame with lines drawn
        """
        result = frame.copy()
        
        if lines is not None:
            color = self.color_schemes[self.color_scheme_idx]
            
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                
                cv2.line(result, (x1, y1), (x2, y2), color, 2)
        
        return result

    def create_display_frame(
        self,
        original: NDArray,
        edges: NDArray,
        lines: NDArray,
        target_width: int = 1400,
        target_height: int = 700
    ) -> NDArray:
        """Create the display frame with all views.

        Args:
            original: Original grayscale frame
            edges: Edge detection result
            lines: Frame with lines drawn
            target_width: Desired width of display
            target_height: Desired height of display

        Returns:
            Combined display frame
        """
        # Convert all to BGR for display
        original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine side by side
        combined = cv2.hconcat([original_bgr, edges_bgr, lines])
        
        # Resize to target dimensions
        return cv2.resize(combined, (target_width, target_height))

    def save_frame(self, frame: NDArray) -> None:
        """Save the current frame.

        Args:
            frame: Frame to save
        """
        edge_method = "canny" if self.use_canny else "sobel"
        filename = self.output_dir / f"frame_{self.frame_count:04d}_{edge_method}.png"
        cv2.imwrite(str(filename), frame)
        logger.info(f"Saved frame to: {filename}")

    def process_video(self, video_path: str) -> bool:
        """Process video and show line detection.

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
            logger.info("  'e' - toggle edge detection method")

            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Remove black area if present
                frame = frame[:, 200:]

                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply histogram equalization
                gray = cv2.equalizeHist(gray)

                # Detect edges
                edges = self.detect_edges(gray)

                # Detect and draw lines
                lines = self.detect_lines(edges)
                lines_frame = self.draw_lines(frame, lines)

                # Create display frame
                display_frame = self.create_display_frame(
                    gray,
                    edges,
                    lines_frame
                )

                # Add parameter info
                edge_method = "Canny" if self.use_canny else "Sobel"
                cv2.putText(
                    display_frame,
                    f"Edge: {edge_method} | Threshold: {self.params.threshold}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    self.color_schemes[self.color_scheme_idx],
                    2
                )

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
                    logger.info(f"Increased sensitivity: {self.params.threshold}")
                elif key == ord('-'):
                    self.params.adjust_sensitivity(increase=False)
                    logger.info(f"Decreased sensitivity: {self.params.threshold}")
                elif key == ord('r'):
                    self.params.reset()
                    logger.info("Reset parameters to default")
                elif key == ord('c'):
                    self.color_scheme_idx = (self.color_scheme_idx + 1) % len(self.color_schemes)
                    logger.info("Changed line color")
                elif key == ord('e'):
                    self.use_canny = not self.use_canny
                    method = "Canny" if self.use_canny else "Sobel"
                    logger.info(f"Switched to {method} edge detection")

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
    """Main function to run line detection visualization.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Check command line arguments
        if len(sys.argv) != 2:
            logger.error("Usage: python 08_hough_transform.py <video_path>")
            return 1

        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return 1

        # Create detector and process video
        detector = LineDetector()
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
