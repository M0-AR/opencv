"""Iterative Image Similarity Detection Module.

This module detects and removes similar images using an iterative approach,
comparing each image with its next neighbor multiple times.

Algorithm Explanation for Beginners:
--------------------------------
The process uses a "sliding window" of size 2 to find similar images:

1. Iterative Processing:
   - Instead of comparing all images at once,
     we compare each image only with its next neighbor
   - This process is repeated multiple times to catch all similar pairs
   - Each iteration may remove some images, making the dataset smaller

2. Neighbor Comparison:
   - For each image:
     * Compare it with the next image in sequence
     * If they are similar, remove the second one
     * Move to the next pair
   - This ensures we process images in order

3. Similarity Check:
   - Convert images to HSV color space (better for color comparison)
   - Create color histograms (like color fingerprints)
   - Calculate distance between histograms
   - If distance < threshold, images are similar

4. Safe Removal:
   - When similar images are found:
     * Move them to backup directory (don't delete)
     * Update image list
     * Continue with next pair

Key Features:
- Iterative neighbor comparison
- Safe image handling (backup instead of delete)
- Progress visualization
- Interactive threshold adjustment
- Result preview
- Detailed logging

Usage:
    python 02_03.py <directory> [--iterations N] [--threshold THRESHOLD]

Controls:
    '+' - Increase threshold
    '-' - Decrease threshold
    's' - Save current results
    'q' - Quit
    Space - Process next pair
"""

from dataclasses import dataclass
import argparse
import logging
from pathlib import Path
import shutil
import sys
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import euclidean
from tqdm import tqdm


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessParams:
    """Parameters for iterative similarity detection."""
    iterations: int = 5
    threshold: float = 0.88
    bins: Tuple[int, int, int] = (8, 8, 8)
    supported_formats: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')
    threshold_step: float = 0.05


class IterativeImageProcessor:
    """A class to handle iterative image similarity detection."""

    def __init__(self, output_dir: str = "processed_images"):
        """Initialize the processor.

        Args:
            output_dir: Base directory for output
        """
        self.output_dir = Path(output_dir)
        self.backup_dir = self.output_dir / "similar_images"
        self.params = ProcessParams()
        self.window_name = "Pair Preview"
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)

    def extract_color_histogram(self, image: NDArray) -> Optional[NDArray]:
        """Extract color histogram from an image.

        Args:
            image: Input image

        Returns:
            Flattened normalized histogram
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate histogram
            hist = cv2.calcHist(
                [hsv],
                [0, 1, 2],
                None,
                self.params.bins,
                [0, 256, 0, 256, 0, 256]
            )
            
            # Normalize
            cv2.normalize(hist, hist)
            
            return hist.flatten()
            
        except Exception as e:
            logger.error(f"Error extracting histogram: {str(e)}")
            return None

    def load_image(self, path: str) -> Optional[NDArray]:
        """Load an image from file.

        Args:
            path: Path to image file

        Returns:
            Loaded image
        """
        try:
            image = cv2.imread(path)
            if image is None:
                raise RuntimeError(f"Could not read image: {path}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None

    def get_image_paths(self, directory: str) -> List[Path]:
        """Get sorted list of image paths.

        Args:
            directory: Directory to search

        Returns:
            List of image paths
        """
        directory = Path(directory)
        
        # Get all image files
        image_paths = []
        for ext in self.params.supported_formats:
            image_paths.extend(directory.glob(f"*{ext}"))
            
        return sorted(image_paths)

    def show_pair_preview(
        self,
        reference_image: NDArray,
        next_image: NDArray
    ) -> None:
        """Show preview of image pair.

        Args:
            reference_image: Reference image
            next_image: Next image in sequence
        """
        try:
            # Resize images
            height = 300
            ref_resized = cv2.resize(
                reference_image,
                (int(height * reference_image.shape[1] / reference_image.shape[0]), height)
            )
            next_resized = cv2.resize(
                next_image,
                (int(height * next_image.shape[1] / next_image.shape[0]), height)
            )
            
            # Create preview
            preview = np.hstack([ref_resized, next_resized])
            
            # Add text
            cv2.putText(
                preview,
                f"Threshold: {self.params.threshold:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            cv2.imshow(self.window_name, preview)
            
        except Exception as e:
            logger.error(f"Error showing preview: {str(e)}")

    def backup_image(self, path: Path) -> None:
        """Move image to backup directory.

        Args:
            path: Path of image to backup
        """
        try:
            dest_path = self.backup_dir / path.name
            shutil.move(path, dest_path)
            logger.info(f"Backed up: {path.name}")
            
        except Exception as e:
            logger.error(f"Error backing up image: {str(e)}")

    def process_directory(self, directory: str) -> bool:
        """Process directory to remove similar images.

        Args:
            directory: Directory containing images

        Returns:
            True if successful
        """
        try:
            # Get image paths
            image_paths = self.get_image_paths(directory)
            if not image_paths:
                logger.error(f"No images found in {directory}")
                return False
                
            logger.info(f"Found {len(image_paths)} images")
            logger.info("Press Space to process current pair")
            logger.info("Press '+'/'-' to adjust threshold")
            logger.info("Press 's' to save current results")
            logger.info("Press 'q' to quit")
            
            total_similar = 0
            
            # Process iterations
            for iteration in range(self.params.iterations):
                logger.info(f"\nIteration {iteration + 1}/{self.params.iterations}")
                
                i = 0
                while i < len(image_paths) - 1:
                    # Load image pair
                    ref_path = image_paths[i]
                    next_path = image_paths[i + 1]
                    
                    ref_image = self.load_image(str(ref_path))
                    next_image = self.load_image(str(next_path))
                    
                    if ref_image is None or next_image is None:
                        i += 1
                        continue
                    
                    # Extract histograms
                    ref_hist = self.extract_color_histogram(ref_image)
                    next_hist = self.extract_color_histogram(next_image)
                    
                    if ref_hist is None or next_hist is None:
                        i += 1
                        continue
                    
                    # Calculate similarity
                    distance = euclidean(ref_hist, next_hist)
                    
                    if distance < self.params.threshold:
                        # Show preview
                        self.show_pair_preview(ref_image, next_image)
                        
                        # Handle input
                        while True:
                            key = cv2.waitKey(0) & 0xFF
                            if key == ord(' '):  # Process
                                self.backup_image(next_path)
                                image_paths.pop(i + 1)
                                total_similar += 1
                                break
                            elif key == ord('+'):  # Increase threshold
                                self.params.threshold += self.params.threshold_step
                                self.show_pair_preview(ref_image, next_image)
                            elif key == ord('-'):  # Decrease threshold
                                self.params.threshold -= self.params.threshold_step
                                self.show_pair_preview(ref_image, next_image)
                            elif key == ord('q'):
                                return True
                    else:
                        i += 1
                
                logger.info(f"Found {total_similar} similar images so far")
            
            logger.info(f"\nProcessed {len(image_paths)} images")
            logger.info(f"Found {total_similar} similar images total")
            return True
            
        except KeyboardInterrupt:
            logger.info("\nProcessing interrupted by user")
            return False
            
        except Exception as e:
            logger.error(f"Error processing directory: {str(e)}")
            return False
            
        finally:
            cv2.destroyAllWindows()


def main() -> int:
    """Main function.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(
            description='Remove similar images using iterative processing'
        )
        parser.add_argument(
            'directory',
            help='Directory containing images'
        )
        parser.add_argument(
            '--iterations',
            type=int,
            default=5,
            help='Number of iterations'
        )
        parser.add_argument(
            '--threshold',
            type=float,
            default=0.88,
            help='Similarity threshold (lower = stricter)'
        )
        
        args = parser.parse_args()
        
        # Create processor
        processor = IterativeImageProcessor()
        processor.params.iterations = args.iterations
        processor.params.threshold = args.threshold
        
        # Process directory
        if processor.process_directory(args.directory):
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