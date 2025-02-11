"""Color Fourier Transform Visualization Module.

A simple and educational module that demonstrates Fourier Transform on color video frames.
It processes each color channel (BGR) separately and shows both the original color image
and its frequency spectrum components side by side.

Key Features:
- Processes each color channel independently
- Shows frequency components for all color channels
- Displays original and transformed images side by side
- Simple keyboard controls for interaction

Usage:
    python 08_fourier_transform_color.py <video_path>

Press 'q' to quit, 's' to save current frame.
Press 'c' to cycle through color channel views:
    - All channels combined
    - Blue channel only
    - Green channel only
    - Red channel only

Date: 2025-02-11
"""

import logging
import os
from pathlib import Path
import sys
from typing import List, Optional, Tuple

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


class ColorFourierTransformer:
    """A simple class to handle color Fourier Transform visualization."""

    def __init__(self, output_dir: str = "fourier_color_output"):
        """Initialize the transformer.

        Args:
            output_dir: Directory to save output images (if requested)
        """
        self.output_dir = Path(output_dir)
        self.frame_count = 0
        self.window_name = 'Original and Color Fourier Transform Spectrum'
        self.display_mode = 0  # 0=all channels, 1=blue, 2=green, 3=red
        self.mode_names = ['All Channels', 'Blue Channel', 'Green Channel', 'Red Channel']
        
        # Create output directory if saving is needed
        self.output_dir.mkdir(exist_ok=True)

    def compute_channel_spectrum(self, channel: NDArray) -> NDArray:
        """Compute the Fourier Transform spectrum of a single color channel.

        Args:
            channel: Single color channel image

        Returns:
            Visualization of the frequency spectrum for this channel
        """
        # Apply Fourier Transform
        fft_channel = fftshift(fft2(channel))
        
        # Compute magnitude spectrum (add 1 to avoid log(0))
        magnitude_spectrum = 20 * np.log(np.abs(fft_channel) + 1)
        
        # Normalize to 0-255 range for visualization
        magnitude_spectrum = cv2.normalize(
            magnitude_spectrum,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )
        
        return cv2.convertScaleAbs(magnitude_spectrum)

    def compute_color_spectrum(self, frame: NDArray) -> NDArray:
        """Compute Fourier Transform spectrum for all color channels.

        Args:
            frame: Input BGR color frame

        Returns:
            Combined spectrum image showing all channels
        """
        # Process each color channel
        spectrum_images = []
        for i in range(3):  # BGR channels
            channel = frame[:, :, i]
            spectrum = self.compute_channel_spectrum(channel)
            spectrum_images.append(spectrum)

        # Handle different display modes
        if self.display_mode == 0:  # All channels
            return cv2.merge(spectrum_images)
        else:  # Single channel
            channel_idx = self.display_mode - 1
            return cv2.merge([
                spectrum_images[channel_idx] if i == channel_idx else 
                np.zeros_like(spectrum_images[0]) 
                for i in range(3)
            ])

    def create_side_by_side_view(
        self,
        frame: NDArray,
        spectrum: NDArray,
        target_width: int = 1400,
        target_height: int = 700
    ) -> NDArray:
        """Create a side-by-side view of original and spectrum images.

        Args:
            frame: Original color frame
            spectrum: Fourier spectrum image
            target_width: Desired width of combined image
            target_height: Desired height of combined image

        Returns:
            Combined image with both views
        """
        # Combine side by side
        combined = cv2.hconcat([frame, spectrum])
        
        # Resize to target dimensions
        return cv2.resize(combined, (target_width, target_height))

    def save_frame(self, frame: NDArray) -> None:
        """Save the current frame.

        Args:
            frame: Frame to save
        """
        mode_name = self.mode_names[self.display_mode].lower().replace(' ', '_')
        filename = self.output_dir / f"frame_{self.frame_count:04d}_{mode_name}.png"
        cv2.imwrite(str(filename), frame)
        logger.info(f"Saved frame to: {filename}")

    def process_video(self, video_path: str) -> bool:
        """Process video and show color Fourier Transform visualization.

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
            logger.info("  'c' - cycle through color channels")

            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Compute color Fourier spectrum
                spectrum = self.compute_color_spectrum(frame)

                # Create side-by-side view
                display_frame = self.create_side_by_side_view(
                    frame,
                    spectrum
                )

                # Add mode indicator
                cv2.putText(
                    display_frame,
                    f"Mode: {self.mode_names[self.display_mode]}",
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
                elif key == ord('c'):
                    self.display_mode = (self.display_mode + 1) % len(self.mode_names)
                    logger.info(f"Switched to: {self.mode_names[self.display_mode]}")

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
    """Main function to run color Fourier Transform visualization.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Check command line arguments
        if len(sys.argv) != 2:
            logger.error("Usage: python 08_fourier_transform_color.py <video_path>")
            return 1

        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return 1

        # Create transformer and process video
        transformer = ColorFourierTransformer()
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
