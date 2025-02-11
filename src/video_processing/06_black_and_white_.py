"""Image Comparison and Difference Detection Module.

This module compares two images and highlights their differences using
structural similarity index (SSIM) and contour detection.

Algorithm Explanation for Beginners:
--------------------------------
We use a step-by-step process to find differences between images:

1. Image Loading and Preprocessing:
   - Load two images to compare
   - Convert them to grayscale
   - Like converting a color photo to black and white
   - This helps focus on structural differences

2. Structural Similarity (SSIM):
   - Compare how similar the images are
   - Like playing "spot the difference"
   - SSIM looks at:
     * Brightness patterns
     * Contrast patterns
     * Structure patterns
   - Gives a score from 0 (different) to 1 (identical)

3. Difference Detection:
   - Create a difference map
   - Like subtracting one image from another
   - White areas show differences
   - Black areas show similarities

4. Contour Finding:
   - Find the outlines of different areas
   - Like drawing boundaries around changes
   - Helps visualize where changes occurred

5. Visualization:
   - Show original and modified images
   - Show difference map
   - Highlight changes in green
   - Makes it easy to spot differences

Key Features:
- Interactive image loading
- Real-time comparison
- Visual difference highlighting
- Progress feedback
- Error handling
- Resource cleanup

Usage:
    python image_compare.py <image1> <image2> [--threshold N]

Controls:
    'n' - Load next image pair
    't' - Adjust threshold
    's' - Save results
    'q' - Quit
"""

from dataclasses import dataclass
import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonParams:
    """Parameters for image comparison."""
    threshold: float = 0.8
    contour_color: Tuple[int, int, int] = (0, 255, 0)
    contour_thickness: int = 3
    min_contour_area: int = 100
    target_size: Tuple[int, int] = (800, 600)
    supported_formats: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')


class ImageComparer:
    """A class to handle image comparison."""

    def __init__(self, output_dir: str = "comparison_output"):
        """Initialize the comparer.

        Args:
            output_dir: Base directory for output
        """
        self.output_dir = Path(output_dir)
        self.params = ComparisonParams()
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize state
        self.image1 = None
        self.image2 = None
        self.diff = None
        self.highlighted = None
        self.score = 0.0
        
        # Set up display
        matplotlib.use('TkAgg')

    def load_images(
        self,
        path1: str,
        path2: str
    ) -> bool:
        """Load image pair.

        Args:
            path1: Path to first image
            path2: Path to second image

        Returns:
            True if successful
        """
        try:
            # Load images
            self.image1 = Image.open(path1)
            self.image2 = Image.open(path2)
            
            # Convert to numpy arrays
            self.image1 = np.array(self.image1)
            self.image2 = np.array(self.image2)
            
            # Resize if needed
            if self.image1.shape != self.image2.shape:
                self.image2 = cv2.resize(
                    self.image2,
                    (self.image1.shape[1], self.image1.shape[0])
                )
            
            logger.info("Images loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading images: {str(e)}")
            return False

    def compare_images(self) -> bool:
        """Compare loaded images.

        Returns:
            True if successful
        """
        try:
            if self.image1 is None or self.image2 is None:
                raise RuntimeError("No images loaded")
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
            
            # Compute SSIM
            score, diff = compare_ssim(gray1, gray2, full=True)
            self.score = score
            self.diff = (diff * 255).astype(np.uint8)
            
            # Find differences
            thresh = cv2.threshold(
                self.diff,
                0,
                255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )[1]
            
            # Find contours
            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Highlight differences
            self.highlighted = self.image2.copy()
            for contour in contours:
                if cv2.contourArea(contour) > self.params.min_contour_area:
                    cv2.drawContours(
                        self.highlighted,
                        [contour],
                        -1,
                        self.params.contour_color,
                        self.params.contour_thickness
                    )
            
            logger.info(f"Comparison complete (SSIM: {score:.3f})")
            return True
            
        except Exception as e:
            logger.error(f"Error comparing images: {str(e)}")
            return False

    def display_results(self) -> None:
        """Display comparison results."""
        try:
            if not all([
                self.image1 is not None,
                self.image2 is not None,
                self.diff is not None,
                self.highlighted is not None
            ]):
                raise RuntimeError("No results to display")
            
            # Create figure
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            
            # Show images
            axs[0, 0].imshow(self.image1)
            axs[0, 0].set_title("Original Image")
            axs[0, 0].axis('off')
            
            axs[0, 1].imshow(self.image2)
            axs[0, 1].set_title("Modified Image")
            axs[0, 1].axis('off')
            
            axs[1, 0].imshow(self.diff, cmap='gray')
            axs[1, 0].set_title("Difference Map")
            axs[1, 0].axis('off')
            
            axs[1, 1].imshow(self.highlighted)
            axs[1, 1].set_title(
                f"Highlighted Differences (SSIM: {self.score:.3f})"
            )
            axs[1, 1].axis('off')
            
            # Show plot
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error displaying results: {str(e)}")

    def save_results(self, prefix: str = "comparison") -> bool:
        """Save comparison results.

        Args:
            prefix: Prefix for output files

        Returns:
            True if successful
        """
        try:
            if not all([
                self.image1 is not None,
                self.image2 is not None,
                self.diff is not None,
                self.highlighted is not None
            ]):
                raise RuntimeError("No results to save")
            
            # Create output paths
            paths = {
                'original': self.output_dir / f"{prefix}_original.jpg",
                'modified': self.output_dir / f"{prefix}_modified.jpg",
                'diff': self.output_dir / f"{prefix}_diff.jpg",
                'highlighted': self.output_dir / f"{prefix}_highlighted.jpg"
            }
            
            # Save images
            cv2.imwrite(str(paths['original']), self.image1)
            cv2.imwrite(str(paths['modified']), self.image2)
            cv2.imwrite(str(paths['diff']), self.diff)
            cv2.imwrite(str(paths['highlighted']), self.highlighted)
            
            logger.info(f"Results saved to {self.output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False

    def run(self, path1: str, path2: str) -> bool:
        """Run comparison workflow.

        Args:
            path1: Path to first image
            path2: Path to second image

        Returns:
            True if successful
        """
        try:
            # Load images
            if not self.load_images(path1, path2):
                return False
            
            # Compare images
            if not self.compare_images():
                return False
            
            # Display results
            self.display_results()
            
            return True
            
        except Exception as e:
            logger.error(f"Error running comparison: {str(e)}")
            return False


def main() -> int:
    """Main function.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(
            description='Compare two images and highlight differences'
        )
        parser.add_argument(
            'image1',
            help='Path to first image'
        )
        parser.add_argument(
            'image2',
            help='Path to second image'
        )
        parser.add_argument(
            '--threshold',
            type=float,
            default=0.8,
            help='SSIM threshold'
        )
        
        args = parser.parse_args()
        
        # Create comparer
        comparer = ImageComparer()
        comparer.params.threshold = args.threshold
        
        # Run comparison
        if not comparer.run(args.image1, args.image2):
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
