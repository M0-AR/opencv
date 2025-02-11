"""Similar Image Removal Module.

This module identifies and removes similar consecutive images from a directory
based on color histogram comparison.

Algorithm Explanation for Beginners:
--------------------------------
The process of removing similar images involves these steps:

1. Color Histograms:
   - Convert each image to HSV color space
     * HSV = Hue (color), Saturation (intensity), Value (brightness)
     * Better than RGB for comparing colors
   - Create a 3D histogram
     * Count how many pixels have each HSV combination
     * Fewer bins = more general comparison
   - Normalize the histogram
     * Make values relative so image size doesn't matter
   
2. Similarity Comparison:
   - For each consecutive pair of images:
     * Compare their histograms using Euclidean distance
     * If distance < threshold, images are similar
     * Skip the second image (mark it for removal)
   - Copy non-skipped images to output directory

The threshold determines how similar images must be:
- Lower threshold = fewer images marked as similar
- Higher threshold = more images marked as similar

Key Features:
- HSV color space for better color comparison
- Configurable similarity threshold
- Progress feedback
- Detailed logging
- Safe copying (no deletion)

Usage:
    python 02_01_removing_similar_images_by_similarity_calculation.py <input_dir> [--threshold THRESHOLD]

Controls:
    Ctrl+C to stop processing
"""

from dataclasses import dataclass
import argparse
import logging
from pathlib import Path
import shutil
import sys
from typing import List, Optional, Set, Tuple

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
class RemovalParams:
    """Parameters for similar image removal."""
    threshold: float = 0.2
    bins: Tuple[int, int, int] = (8, 8, 8)
    supported_formats: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')


class SimilarImageRemover:
    """A class to handle similar image removal."""

    def __init__(self, output_dir: str = "reduced_frames"):
        """Initialize the remover.

        Args:
            output_dir: Directory to save non-similar images
        """
        self.output_dir = Path(output_dir)
        self.params = RemovalParams()
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

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

    def find_similar_images(self, directory: str) -> Optional[Set[Path]]:
        """Find similar consecutive images.

        Args:
            directory: Directory containing images

        Returns:
            Set of paths to similar images
        """
        try:
            # Get image paths
            image_paths = self.get_image_paths(directory)
            if not image_paths:
                logger.error(f"No images found in {directory}")
                return None
                
            logger.info(f"Found {len(image_paths)} images")
            
            # Find similar images
            similar_images = set()
            
            for i in tqdm(range(len(image_paths) - 1), desc="Finding similar images"):
                # Skip if current image was marked as similar
                if image_paths[i] in similar_images:
                    continue
                    
                # Load and process current image
                current_image = self.load_image(str(image_paths[i]))
                if current_image is None:
                    continue
                    
                current_hist = self.extract_color_histogram(current_image)
                if current_hist is None:
                    continue
                
                # Load and process next image
                next_image = self.load_image(str(image_paths[i + 1]))
                if next_image is None:
                    continue
                    
                next_hist = self.extract_color_histogram(next_image)
                if next_hist is None:
                    continue
                
                # Compare histograms
                distance = euclidean(current_hist, next_hist)
                
                # Mark as similar if below threshold
                if distance < self.params.threshold:
                    similar_images.add(image_paths[i + 1])
                    logger.debug(
                        f"Similar images found:\n"
                        f"  {image_paths[i]}\n"
                        f"  {image_paths[i + 1]}\n"
                        f"  Distance: {distance:.4f}"
                    )
            
            return similar_images
            
        except Exception as e:
            logger.error(f"Error finding similar images: {str(e)}")
            return None

    def copy_unique_images(
        self,
        directory: str,
        similar_images: Set[Path]
    ) -> bool:
        """Copy non-similar images to output directory.

        Args:
            directory: Source directory
            similar_images: Set of similar image paths to skip

        Returns:
            True if successful
        """
        try:
            # Get all image paths
            image_paths = self.get_image_paths(directory)
            
            # Copy non-similar images
            copied_count = 0
            skipped_count = 0
            
            for path in tqdm(image_paths, desc="Copying unique images"):
                if path in similar_images:
                    skipped_count += 1
                    continue
                    
                # Copy image
                dest_path = self.output_dir / path.name
                shutil.copy2(path, dest_path)
                copied_count += 1
            
            logger.info(
                f"Copied {copied_count} unique images, "
                f"skipped {skipped_count} similar images"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error copying images: {str(e)}")
            return False

    def process_directory(self, directory: str) -> bool:
        """Process directory to remove similar images.

        Args:
            directory: Directory containing images

        Returns:
            True if successful
        """
        try:
            # Find similar images
            similar_images = self.find_similar_images(directory)
            if similar_images is None:
                return False
                
            # Copy unique images
            return self.copy_unique_images(directory, similar_images)
            
        except KeyboardInterrupt:
            logger.info("\nProcessing interrupted by user")
            return False
            
        except Exception as e:
            logger.error(f"Error processing directory: {str(e)}")
            return False


def main() -> int:
    """Main function.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(
            description='Remove similar consecutive images from directory'
        )
        parser.add_argument(
            'directory',
            help='Directory containing images'
        )
        parser.add_argument(
            '--threshold',
            type=float,
            default=0.2,
            help='Similarity threshold (lower = stricter)'
        )
        parser.add_argument(
            '--bins',
            type=int,
            nargs=3,
            default=[8, 8, 8],
            help='Number of histogram bins for each channel'
        )
        
        args = parser.parse_args()
        
        # Create remover
        remover = SimilarImageRemover()
        remover.params.threshold = args.threshold
        remover.params.bins = tuple(args.bins)
        
        # Process directory
        if remover.process_directory(args.directory):
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