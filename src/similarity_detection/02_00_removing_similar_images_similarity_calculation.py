"""Image Similarity Detection Module.

This module calculates similarity scores between consecutive images in a directory
using color histogram comparison.

Algorithm Explanation for Beginners:
--------------------------------
The process of finding similar images involves these steps:

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
   - For each image:
     * Compare its histogram with previous/next images
     * Use Euclidean distance (like measuring a straight line)
     * Smaller distance = more similar images
   - Save results to a file for review

Key Features:
- HSV color space for better color comparison
- Configurable histogram bins
- Progress feedback
- Detailed logging
- Result saving

Usage:
    python 02_00_removing_similar_images_similarity_calculation.py <directory> [--bins BINS]

Controls:
    Ctrl+C to stop processing
"""

from dataclasses import dataclass
import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

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
class SimilarityParams:
    """Parameters for similarity calculation."""
    bins: Tuple[int, int, int] = (8, 8, 8)
    supported_formats: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')


class ImageSimilarityCalculator:
    """A class to handle image similarity calculations."""

    def __init__(self, output_dir: str = "similarity_output"):
        """Initialize the calculator.

        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.params = SimilarityParams()
        
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

    def calculate_similarities(
        self,
        directory: str
    ) -> Optional[Dict[str, List[Tuple[str, float]]]]:
        """Calculate similarities between consecutive images.

        Args:
            directory: Directory containing images

        Returns:
            Dictionary mapping image paths to lists of (compared path, score)
        """
        try:
            # Get image paths
            image_paths = self.get_image_paths(directory)
            if not image_paths:
                logger.error(f"No images found in {directory}")
                return None
                
            logger.info(f"Found {len(image_paths)} images")
            
            # Calculate similarities
            similarities: Dict[str, List[Tuple[str, float]]] = {}
            
            for i, path in enumerate(tqdm(image_paths, desc="Calculating similarities")):
                # Load and process current image
                current_image = self.load_image(str(path))
                if current_image is None:
                    continue
                    
                current_hist = self.extract_color_histogram(current_image)
                if current_hist is None:
                    continue
                
                similarities[str(path)] = []
                
                # Compare with previous image
                if i > 0:
                    prev_path = image_paths[i - 1]
                    prev_image = self.load_image(str(prev_path))
                    if prev_image is not None:
                        prev_hist = self.extract_color_histogram(prev_image)
                        if prev_hist is not None:
                            distance = euclidean(current_hist, prev_hist)
                            similarities[str(path)].append(
                                (str(prev_path), distance)
                            )
                
                # Compare with next image
                if i < len(image_paths) - 1:
                    next_path = image_paths[i + 1]
                    next_image = self.load_image(str(next_path))
                    if next_image is not None:
                        next_hist = self.extract_color_histogram(next_image)
                        if next_hist is not None:
                            distance = euclidean(current_hist, next_hist)
                            similarities[str(path)].append(
                                (str(next_path), distance)
                            )
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error calculating similarities: {str(e)}")
            return None

    def save_results(
        self,
        similarities: Dict[str, List[Tuple[str, float]]]
    ) -> bool:
        """Save similarity results to file.

        Args:
            similarities: Dictionary of similarity scores

        Returns:
            True if successful
        """
        try:
            output_file = self.output_dir / "similarity_scores.txt"
            
            with open(output_file, 'w') as f:
                for current_path, scores in similarities.items():
                    f.write(f"Current Image: {current_path}\n")
                    for compare_path, score in scores:
                        f.write(
                            f"    Compared to {compare_path}, "
                            f"Score: {score:.4f}\n"
                        )
                    f.write("\n")
                    
            logger.info(f"Results saved to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False

    def process_directory(self, directory: str) -> bool:
        """Process all images in a directory.

        Args:
            directory: Directory containing images

        Returns:
            True if successful
        """
        try:
            # Calculate similarities
            similarities = self.calculate_similarities(directory)
            if similarities is None:
                return False
                
            # Save results
            return self.save_results(similarities)
            
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
            description='Calculate image similarities in directory'
        )
        parser.add_argument(
            'directory',
            help='Directory containing images'
        )
        parser.add_argument(
            '--bins',
            type=int,
            nargs=3,
            default=[8, 8, 8],
            help='Number of histogram bins for each channel'
        )
        
        args = parser.parse_args()
        
        # Create calculator
        calculator = ImageSimilarityCalculator()
        calculator.params.bins = tuple(args.bins)
        
        # Process directory
        if calculator.process_directory(args.directory):
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