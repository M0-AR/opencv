"""Principal Component Analysis (PCA) Image Processing Module.

This module demonstrates real-time dimensionality reduction using PCA on video frames.
It provides a visual comparison between the original frame and its PCA reconstruction.

Algorithm Explanation for Beginners:
--------------------------------
PCA (Principal Component Analysis) is a technique that helps us simplify complex data
while keeping the most important information. Think of it like summarizing a detailed
story into its main points. Here's how it works:

1. Data Preparation:
   - We take each frame and convert it to grayscale (black and white)
   - We "flatten" the 2D image into a long list of pixel values
   
2. Finding Patterns (Principal Components):
   - PCA looks for the most important patterns in the pixel values
   - These patterns are like building blocks that can reconstruct the image
   - We keep only the most important patterns (controlled by n_components)
   
3. Reconstruction:
   - We use these patterns to rebuild a simplified version of the image
   - This shows us what information PCA considers most important

The result shows:
- Left: Original grayscale image
- Right: Reconstructed image using only the most important components

This helps us understand what features PCA considers most significant in the image.

Key Features:
- Real-time PCA analysis
- Adjustable compression ratio
- Side-by-side comparison
- Interactive parameter tuning
- Frame saving capability

Usage:
    python 08_pca.py <video_path>

Controls:
    'q' - quit
    's' - save current frame
    '+' - increase components (less compression)
    '-' - decrease components (more compression)
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
from sklearn.decomposition import PCA


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PCAParams:
    """Parameters for PCA processing."""
    n_components: float = 0.5  # Explained variance ratio (0-1)
    min_components: float = 0.1
    max_components: float = 0.9
    use_histogram: bool = False
    
    def adjust_components(self, increase: bool = True) -> None:
        """Adjust number of components.
        
        Args:
            increase: If True, increase components; if False, decrease
        """
        step = 0.1
        if increase and self.n_components < self.max_components:
            self.n_components = min(self.n_components + step, self.max_components)
        elif not increase and self.n_components > self.min_components:
            self.n_components = max(self.n_components - step, self.min_components)
            
    def reset(self) -> None:
        """Reset parameters to default values."""
        self.__init__()


class PCAProcessor:
    """A class to handle PCA processing and visualization."""

    def __init__(self, output_dir: str = "pca_output"):
        """Initialize the processor.

        Args:
            output_dir: Directory to save output images
        """
        self.output_dir = Path(output_dir)
        self.frame_count = 0
        self.window_name = 'Original and PCA Results'
        self.params = PCAParams()
        
        # Create output directory if needed
        self.output_dir.mkdir(exist_ok=True)

    def preprocess_frame(self, frame: NDArray) -> NDArray:
        """Preprocess frame before PCA.

        Args:
            frame: Input frame

        Returns:
            Preprocessed grayscale frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization if enabled
        if self.params.use_histogram:
            gray = cv2.equalizeHist(gray)
            
        return gray

    def apply_pca(self, frame: NDArray) -> NDArray:
        """Apply PCA to frame.

        Args:
            frame: Input grayscale frame

        Returns:
            PCA reconstructed frame
        """
        # Reshape to 2D array
        h, w = frame.shape
        pixels = frame.reshape(-1, 1)
        
        # Apply PCA
        pca = PCA(n_components=self.params.n_components)
        pca.fit(pixels)
        
        # Transform and reconstruct
        transformed = pca.transform(pixels)
        reconstructed = pca.inverse_transform(transformed)
        
        # Reshape back to image
        return reconstructed.reshape(h, w)

    def create_display_frame(
        self,
        original: NDArray,
        pca_result: NDArray,
        target_width: int = 1400,
        target_height: int = 700
    ) -> NDArray:
        """Create display frame with both views.

        Args:
            original: Original frame
            pca_result: PCA processed frame
            target_width: Desired width
            target_height: Desired height

        Returns:
            Combined display frame
        """
        # Convert PCA result to uint8
        pca_image = np.uint8(pca_result)
        
        # Convert both to BGR for display
        original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        pca_bgr = cv2.cvtColor(pca_image, cv2.COLOR_GRAY2BGR)
        
        # Combine side by side
        combined = cv2.hconcat([original_bgr, pca_bgr])
        
        # Resize to target dimensions
        return cv2.resize(combined, (target_width, target_height))

    def save_frame(self, frame: NDArray) -> None:
        """Save current frame.

        Args:
            frame: Frame to save
        """
        hist = "hist" if self.params.use_histogram else "nohist"
        filename = self.output_dir / f"frame_{self.frame_count:04d}_{hist}.png"
        cv2.imwrite(str(filename), frame)
        logger.info(f"Saved frame to: {filename}")

    def process_video(self, video_path: str) -> bool:
        """Process video with PCA visualization.

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
            logger.info("  '+' - increase components (less compression)")
            logger.info("  '-' - decrease components (more compression)")
            logger.info("  'r' - reset parameters")
            logger.info("  'h' - toggle histogram equalization")

            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                gray = self.preprocess_frame(frame)
                pca_result = self.apply_pca(gray)
                
                # Create display frame
                display_frame = self.create_display_frame(gray, pca_result)

                # Add parameter info
                hist = "On" if self.params.use_histogram else "Off"
                cv2.putText(
                    display_frame,
                    f"Components: {self.params.n_components:.1f} | Hist: {hist}",
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
                    self.params.adjust_components(increase=True)
                    logger.info(f"Increased components to: {self.params.n_components:.1f}")
                elif key == ord('-'):
                    self.params.adjust_components(increase=False)
                    logger.info(f"Decreased components to: {self.params.n_components:.1f}")
                elif key == ord('r'):
                    self.params.reset()
                    logger.info("Reset parameters to default")
                elif key == ord('h'):
                    self.params.use_histogram = not self.params.use_histogram
                    hist = "enabled" if self.params.use_histogram else "disabled"
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
    """Main function.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Check command line arguments
        if len(sys.argv) != 2:
            logger.error("Usage: python 08_pca.py <video_path>")
            return 1

        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return 1

        # Create processor and process video
        processor = PCAProcessor()
        if processor.process_video(video_path):
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
