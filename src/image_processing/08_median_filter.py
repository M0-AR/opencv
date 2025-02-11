"""Median Filter Image Processing Module.

This module demonstrates real-time median filtering for noise reduction in video frames.
It shows the original frame and filtered result side by side, allowing interactive
adjustment of filter parameters for optimal noise reduction.

Key Features:
- Real-time median filtering
- Adjustable kernel size
- Interactive parameter tuning
- Multiple color modes (RGB/Grayscale)
- Side-by-side view comparison
- Simple keyboard controls for interaction

Usage:
    python 08_median_filter.py <video_path>

Controls:
    'q' - quit
    's' - save current frame
    '+' - increase kernel size
    '-' - decrease kernel size
    'r' - reset parameters to default
    'c' - toggle color/grayscale mode
    'h' - toggle histogram equalization

Date: 2025-02-11
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


# Set up logging with a simple format
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format for better readability
)
logger = logging.getLogger(__name__)


@dataclass
class FilterParams:
    """Parameters for median filtering."""
    kernel_size: int = 5  # Must be odd
    min_kernel: int = 3
    max_kernel: int = 15
    
    def adjust_kernel(self, increase: bool = True) -> None:
        """Adjust kernel size.
        
        Args:
            increase: If True, increase size; if False, decrease
        """
        if increase and self.kernel_size < self.max_kernel:
            self.kernel_size = min(self.kernel_size + 2, self.max_kernel)
        elif not increase and self.kernel_size > self.min_kernel:
            self.kernel_size = max(self.kernel_size - 2, self.min_kernel)
            
    def reset(self) -> None:
        """Reset parameters to default values."""
        self.__init__()


class MedianFilter:
    """A class to handle median filtering visualization."""

    def __init__(self, output_dir: str = "median_filter_output"):
        """Initialize the filter.

        Args:
            output_dir: Directory to save output images (if requested)
        """
        self.output_dir = Path(output_dir)
        self.frame_count = 0
        self.window_name = 'Original and Median Filtered Frame'
        self.params = FilterParams()
        self.use_color = True
        self.use_histogram = True
        
        # Create output directory if saving is needed
        self.output_dir.mkdir(exist_ok=True)

    def apply_median_filter(self, frame: NDArray) -> NDArray:
        """Apply median filter to frame.

        Args:
            frame: Input frame

        Returns:
            Filtered frame
        """
        return cv2.medianBlur(frame, self.params.kernel_size)

    def preprocess_frame(self, frame: NDArray) -> NDArray:
        """Preprocess frame before filtering.

        Args:
            frame: Input frame

        Returns:
            Preprocessed frame
        """
        if not self.use_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        if self.use_histogram:
            if self.use_color:
                # Convert to LAB for better color histogram equalization
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = cv2.equalizeHist(l)
                lab = cv2.merge([l, a, b])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                frame = cv2.equalizeHist(frame)
                
        return frame

    def create_display_frame(
        self,
        original: NDArray,
        filtered: NDArray,
        target_width: int = 1400,
        target_height: int = 700
    ) -> NDArray:
        """Create the display frame with both views.

        Args:
            original: Original frame
            filtered: Filtered frame
            target_width: Desired width of display
            target_height: Desired height of display

        Returns:
            Combined display frame
        """
        # Convert to BGR for display if in grayscale
        if not self.use_color:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
            filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        
        # Combine side by side
        combined = cv2.hconcat([original, filtered])
        
        # Resize to target dimensions
        return cv2.resize(combined, (target_width, target_height))

    def save_frame(self, frame: NDArray) -> None:
        """Save the current frame.

        Args:
            frame: Frame to save
        """
        mode = "color" if self.use_color else "gray"
        hist = "hist" if self.use_histogram else "nohist"
        filename = self.output_dir / f"frame_{self.frame_count:04d}_{mode}_{hist}.png"
        cv2.imwrite(str(filename), frame)
        logger.info(f"Saved frame to: {filename}")

    def process_video(self, video_path: str) -> bool:
        """Process video and show filtering results.

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
            logger.info("  '+' - increase kernel size")
            logger.info("  '-' - decrease kernel size")
            logger.info("  'r' - reset parameters")
            logger.info("  'c' - toggle color/grayscale")
            logger.info("  'h' - toggle histogram equalization")

            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Preprocess frame
                processed = self.preprocess_frame(frame)

                # Apply median filter
                filtered = self.apply_median_filter(processed)

                # Create display frame
                display_frame = self.create_display_frame(processed, filtered)

                # Add parameter info
                mode = "Color" if self.use_color else "Grayscale"
                hist = "On" if self.use_histogram else "Off"
                cv2.putText(
                    display_frame,
                    f"Kernel: {self.params.kernel_size}x{self.params.kernel_size} | "
                    f"Mode: {mode} | Hist: {hist}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
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
                    self.params.adjust_kernel(increase=True)
                    logger.info(f"Increased kernel size to: {self.params.kernel_size}")
                elif key == ord('-'):
                    self.params.adjust_kernel(increase=False)
                    logger.info(f"Decreased kernel size to: {self.params.kernel_size}")
                elif key == ord('r'):
                    self.params.reset()
                    logger.info("Reset parameters to default")
                elif key == ord('c'):
                    self.use_color = not self.use_color
                    mode = "Color" if self.use_color else "Grayscale"
                    logger.info(f"Switched to {mode} mode")
                elif key == ord('h'):
                    self.use_histogram = not self.use_histogram
                    hist = "enabled" if self.use_histogram else "disabled"
                    logger.info(f"Histogram equalization {hist}")

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
    """Main function to run median filter visualization.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Check command line arguments
        if len(sys.argv) != 2:
            logger.error("Usage: python 08_median_filter.py <video_path>")
            return 1

        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return 1

        # Create filter and process video
        filter = MedianFilter()
        if filter.process_video(video_path):
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
