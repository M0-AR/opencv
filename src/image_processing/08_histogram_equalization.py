"""Histogram Equalization Visualization Module.

A simple and educational module that demonstrates histogram equalization on video frames.
It shows both the original grayscale image and its histogram-equalized version side by side,
helping users understand how histogram equalization enhances image contrast.

Key Features:
- Converts video frames to grayscale
- Applies histogram equalization
- Shows original and enhanced images side by side
- Displays live histograms (optional)
- Simple keyboard controls for interaction

Usage:
    python 08_histogram_equalization.py <video_path>

Controls:
    'q' - quit
    's' - save current frame
    'h' - toggle histogram display
    'c' - toggle color/grayscale mode

Date: 2025-02-11
"""

import logging
import os
from pathlib import Path
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


# Set up logging with a simple format
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format for better readability
)
logger = logging.getLogger(__name__)


class HistogramEqualizer:
    """A simple class to handle histogram equalization visualization."""

    def __init__(self, output_dir: str = "histogram_output"):
        """Initialize the equalizer.

        Args:
            output_dir: Directory to save output images (if requested)
        """
        self.output_dir = Path(output_dir)
        self.frame_count = 0
        self.window_name = 'Original and Equalized Frames'
        self.show_histogram = False
        self.color_mode = False  # False = grayscale, True = color
        
        # Create output directory if saving is needed
        self.output_dir.mkdir(exist_ok=True)

    def compute_histogram(self, image: NDArray) -> NDArray:
        """Compute histogram for visualization.

        Args:
            image: Input grayscale image

        Returns:
            Histogram visualization image
        """
        # Compute histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
        # Create histogram visualization
        hist_h = 200
        hist_w = 256
        hist_image = np.zeros((hist_h, hist_w), dtype=np.uint8)
        
        # Normalize histogram for visualization
        cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
        
        # Draw histogram
        for i in range(256):
            cv2.line(
                hist_image,
                (i, hist_h),
                (i, hist_h - int(hist[i])),
                255,
                1
            )
            
        return hist_image

    def equalize_image(self, frame: NDArray) -> Tuple[NDArray, NDArray]:
        """Apply histogram equalization to an image.

        Args:
            frame: Input frame (BGR or grayscale)

        Returns:
            Tuple of (original_frame, equalized_frame) in the same format
        """
        if self.color_mode:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Equalize L channel
            l_eq = cv2.equalizeHist(l)
            
            # Merge channels and convert back
            lab_eq = cv2.merge([l_eq, a, b])
            equalized = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
            
            return frame, equalized
        else:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            
            # Convert both to BGR for display
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            
            return gray_bgr, equalized_bgr

    def create_display_frame(
        self,
        original: NDArray,
        equalized: NDArray,
        target_width: int = 1400,
        target_height: int = 700
    ) -> NDArray:
        """Create the display frame with optional histograms.

        Args:
            original: Original frame
            equalized: Equalized frame
            target_width: Desired width of display
            target_height: Desired height of display

        Returns:
            Combined display frame
        """
        if self.show_histogram:
            # Compute histograms
            if self.color_mode:
                # Use luminance for histogram in color mode
                orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                eq_gray = cv2.cvtColor(equalized, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                eq_gray = cv2.cvtColor(equalized, cv2.COLOR_BGR2GRAY)
                
            hist_orig = self.compute_histogram(orig_gray)
            hist_eq = self.compute_histogram(eq_gray)
            
            # Convert histograms to BGR
            hist_orig_bgr = cv2.cvtColor(hist_orig, cv2.COLOR_GRAY2BGR)
            hist_eq_bgr = cv2.cvtColor(hist_eq, cv2.COLOR_GRAY2BGR)
            
            # Stack images and histograms
            left_side = np.vstack([original, hist_orig_bgr])
            right_side = np.vstack([equalized, hist_eq_bgr])
            combined = cv2.hconcat([left_side, right_side])
        else:
            # Just combine images side by side
            combined = cv2.hconcat([original, equalized])
        
        # Resize to target dimensions
        return cv2.resize(combined, (target_width, target_height))

    def save_frame(self, frame: NDArray) -> None:
        """Save the current frame.

        Args:
            frame: Frame to save
        """
        mode = "color" if self.color_mode else "gray"
        hist = "with_hist" if self.show_histogram else "no_hist"
        filename = self.output_dir / f"frame_{self.frame_count:04d}_{mode}_{hist}.png"
        cv2.imwrite(str(filename), frame)
        logger.info(f"Saved frame to: {filename}")

    def process_video(self, video_path: str) -> bool:
        """Process video and show histogram equalization.

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
            logger.info("  'h' - toggle histogram display")
            logger.info("  'c' - toggle color/grayscale mode")

            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Apply equalization
                original, equalized = self.equalize_image(frame)

                # Create display frame
                display_frame = self.create_display_frame(
                    original,
                    equalized
                )

                # Add mode indicator
                mode_text = f"Mode: {'Color' if self.color_mode else 'Grayscale'}"
                hist_text = " | Histogram: On" if self.show_histogram else ""
                cv2.putText(
                    display_frame,
                    mode_text + hist_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
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
                elif key == ord('h'):
                    self.show_histogram = not self.show_histogram
                    logger.info(f"Histogram display: {'On' if self.show_histogram else 'Off'}")
                elif key == ord('c'):
                    self.color_mode = not self.color_mode
                    logger.info(f"Switched to: {'Color' if self.color_mode else 'Grayscale'} mode")

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
    """Main function to run histogram equalization visualization.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Check command line arguments
        if len(sys.argv) != 2:
            logger.error("Usage: python 08_histogram_equalization.py <video_path>")
            return 1

        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return 1

        # Create equalizer and process video
        equalizer = HistogramEqualizer()
        if equalizer.process_video(video_path):
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
