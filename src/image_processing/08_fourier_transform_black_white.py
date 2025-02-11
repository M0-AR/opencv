"""Fourier Transform Visualization Module.

A simple and educational module that demonstrates Fourier Transform on video frames.
It shows both the original grayscale image and its frequency spectrum side by side.

Key Features:
- Converts video frames to grayscale
- Applies Fourier Transform to visualize frequency components
- Shows original and transformed images side by side
- Simple keyboard controls for interaction

Usage:
    python 08_fourier_transform_black_white.py <video_path>

Press 'q' to quit, 's' to save current frame.

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
from scipy.fftpack import fft2, fftshift


# Set up logging with a simple format
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format for better readability
)
logger = logging.getLogger(__name__)


class FourierTransformer:
    """A simple class to handle Fourier Transform visualization."""

    def __init__(self, output_dir: str = "fourier_output"):
        """Initialize the transformer.

        Args:
            output_dir: Directory to save output images (if requested)
        """
        self.output_dir = Path(output_dir)
        self.frame_count = 0
        self.window_name = 'Original and Fourier Transform Spectrum'
        
        # Create output directory if saving is needed
        self.output_dir.mkdir(exist_ok=True)

    def compute_fourier_spectrum(self, gray_frame: NDArray) -> NDArray:
        """Compute the Fourier Transform spectrum of a grayscale image.

        Args:
            gray_frame: Input grayscale image

        Returns:
            Visualization of the frequency spectrum
        """
        # Apply Fourier Transform
        fft_frame = fftshift(fft2(gray_frame))
        
        # Compute magnitude spectrum (add 1 to avoid log(0))
        magnitude_spectrum = 20 * np.log(np.abs(fft_frame) + 1)
        
        # Normalize to 0-255 range for visualization
        magnitude_spectrum = cv2.normalize(
            magnitude_spectrum,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )
        
        return cv2.convertScaleAbs(magnitude_spectrum)

    def create_side_by_side_view(
        self,
        gray_frame: NDArray,
        spectrum: NDArray,
        target_width: int = 1400,
        target_height: int = 700
    ) -> NDArray:
        """Create a side-by-side view of original and spectrum images.

        Args:
            gray_frame: Original grayscale frame
            spectrum: Fourier spectrum image
            target_width: Desired width of combined image
            target_height: Desired height of combined image

        Returns:
            Combined image with both views
        """
        # Convert both to BGR for consistent display
        gray_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        spectrum_bgr = cv2.cvtColor(spectrum, cv2.COLOR_GRAY2BGR)
        
        # Combine side by side
        combined = cv2.hconcat([gray_bgr, spectrum_bgr])
        
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
        """Process video and show Fourier Transform visualization.

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
            logger.info("Press 'q' to quit, 's' to save current frame")

            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Compute Fourier spectrum
                spectrum = self.compute_fourier_spectrum(gray_frame)

                # Create side-by-side view
                display_frame = self.create_side_by_side_view(
                    gray_frame,
                    spectrum
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
    """Main function to run Fourier Transform visualization.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Check command line arguments
        if len(sys.argv) != 2:
            logger.error("Usage: python 08_fourier_transform_black_white.py <video_path>")
            return 1

        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return 1

        # Create transformer and process video
        transformer = FourierTransformer()
        if transformer.process_video(video_path):
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
