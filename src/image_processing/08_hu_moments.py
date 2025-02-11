"""Hu Moments Shape Analysis Module.

This module demonstrates shape analysis using Hu Moments in video frames.
It shows the original frame and detected contours side by side, while
calculating and displaying Hu Moments for significant contours.

Key Features:
- Real-time shape analysis
- Contour detection and visualization
- Hu Moments calculation
- Adjustable detection parameters
- Interactive parameter tuning
- Shape visualization with configurable colors
- Simple keyboard controls for interaction

Usage:
    python 08_hu_moments.py <video_path>

Controls:
    'q' - quit
    's' - save current frame
    '+' - increase detection sensitivity
    '-' - decrease detection sensitivity
    'r' - reset parameters to default
    'c' - cycle through color schemes
    'h' - toggle Hu Moments display

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
class ShapeDetectionParams:
    """Parameters for shape detection and analysis."""
    canny_low: int = 50  # Lower threshold for Canny
    canny_high: int = 150  # Upper threshold for Canny
    min_contour_area: float = 100.0  # Minimum contour area to analyze
    max_contour_area: float = 10000.0  # Maximum contour area to analyze
    
    def adjust_sensitivity(self, increase: bool = True) -> None:
        """Adjust detection sensitivity.
        
        Args:
            increase: If True, increase sensitivity; if False, decrease
        """
        factor = 0.9 if increase else 1.1  # Inverse for thresholds
        self.canny_low = int(self.canny_low * factor)
        self.canny_high = int(self.canny_high * factor)
        self.min_contour_area *= factor
        self.max_contour_area /= factor
        
    def reset(self) -> None:
        """Reset parameters to default values."""
        self.__init__()


class ShapeAnalyzer:
    """A class to handle shape analysis and visualization."""

    def __init__(self, output_dir: str = "shape_analysis_output"):
        """Initialize the analyzer.

        Args:
            output_dir: Directory to save output images (if requested)
        """
        self.output_dir = Path(output_dir)
        self.frame_count = 0
        self.window_name = 'Original and Contours with Hu Moments'
        self.params = ShapeDetectionParams()
        self.color_scheme_idx = 0
        self.show_hu_moments = True
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
        return cv2.Canny(
            frame,
            self.params.canny_low,
            self.params.canny_high
        )

    def find_contours(self, edges: NDArray) -> Tuple[List[NDArray], NDArray]:
        """Find contours in the edge image.

        Args:
            edges: Edge detection result

        Returns:
            Tuple of (filtered contours, contour image)
        """
        # Find all contours
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area
        filtered_contours = []
        contour_img = np.zeros_like(edges)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.params.min_contour_area <= area <= self.params.max_contour_area:
                filtered_contours.append(contour)
                
        return filtered_contours, contour_img

    def calculate_hu_moments(self, contour: NDArray) -> NDArray:
        """Calculate Hu Moments for a contour.

        Args:
            contour: Input contour

        Returns:
            Array of Hu Moments
        """
        moments = cv2.moments(contour)
        return cv2.HuMoments(moments)

    def draw_contours(
        self,
        image: NDArray,
        contours: List[NDArray],
        draw_hu: bool = True
    ) -> NDArray:
        """Draw contours and optionally Hu Moments.

        Args:
            image: Input image to draw on
            contours: List of contours to draw
            draw_hu: Whether to draw Hu Moments

        Returns:
            Image with contours and optionally Hu Moments drawn
        """
        result = image.copy()
        color = self.color_schemes[self.color_scheme_idx]
        
        for i, contour in enumerate(contours):
            # Draw contour
            cv2.drawContours(result, [contour], -1, color, 2)
            
            if draw_hu and self.show_hu_moments:
                # Calculate and draw Hu Moments
                hu_moments = self.calculate_hu_moments(contour)
                
                # Get contour center for text placement
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Format first 3 Hu Moments (most significant)
                    text = f"Hu: {hu_moments[0][0]:.2e}"
                    cv2.putText(
                        result,
                        text,
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    )
        
        return result

    def create_display_frame(
        self,
        original: NDArray,
        contours: List[NDArray],
        target_width: int = 1400,
        target_height: int = 700
    ) -> NDArray:
        """Create the display frame with all views.

        Args:
            original: Original grayscale frame
            contours: List of detected contours
            target_width: Desired width of display
            target_height: Desired height of display

        Returns:
            Combined display frame
        """
        # Create contour visualization
        contour_img = np.zeros_like(original)
        contour_display = self.draw_contours(contour_img, contours)
        
        # Convert to BGR for display
        original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        contour_bgr = cv2.cvtColor(contour_display, cv2.COLOR_GRAY2BGR)
        
        # Combine side by side
        combined = cv2.hconcat([original_bgr, contour_bgr])
        
        # Resize to target dimensions
        return cv2.resize(combined, (target_width, target_height))

    def save_frame(self, frame: NDArray) -> None:
        """Save the current frame.

        Args:
            frame: Frame to save
        """
        filename = self.output_dir / f"frame_{self.frame_count:04d}.png"
        cv2.imwrite(str(filename), frame)
        logger.info(f"Saved frame to: {filename}")

    def process_video(self, video_path: str) -> bool:
        """Process video and show shape analysis.

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
            logger.info("  'h' - toggle Hu Moments display")

            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply histogram equalization
                gray = cv2.equalizeHist(gray)

                # Detect edges
                edges = self.detect_edges(gray)

                # Find contours
                contours, _ = self.find_contours(edges)

                # Create display frame
                display_frame = self.create_display_frame(gray, contours)

                # Add parameter info
                cv2.putText(
                    display_frame,
                    f"Threshold: {self.params.canny_low}-{self.params.canny_high} | "
                    f"Hu Moments: {'On' if self.show_hu_moments else 'Off'}",
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
                    logger.info(f"Increased sensitivity: {self.params.canny_low}-{self.params.canny_high}")
                elif key == ord('-'):
                    self.params.adjust_sensitivity(increase=False)
                    logger.info(f"Decreased sensitivity: {self.params.canny_low}-{self.params.canny_high}")
                elif key == ord('r'):
                    self.params.reset()
                    logger.info("Reset parameters to default")
                elif key == ord('c'):
                    self.color_scheme_idx = (self.color_scheme_idx + 1) % len(self.color_schemes)
                    logger.info("Changed contour color")
                elif key == ord('h'):
                    self.show_hu_moments = not self.show_hu_moments
                    logger.info(f"Hu Moments display: {'On' if self.show_hu_moments else 'Off'}")

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
    """Main function to run shape analysis visualization.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Check command line arguments
        if len(sys.argv) != 2:
            logger.error("Usage: python 08_hu_moments.py <video_path>")
            return 1

        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return 1

        # Create analyzer and process video
        analyzer = ShapeAnalyzer()
        if analyzer.process_video(video_path):
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
