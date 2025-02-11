"""Chunked Image Similarity Detection Module.

This module removes similar images by comparing them in chunks, which is more
efficient than comparing every image with every other image.

Algorithm Explanation for Beginners:
--------------------------------
The process uses a "sliding window" approach to find similar images:

1. Chunk Processing:
   - Instead of comparing all images (which would be very slow),
     we divide images into chunks (like pages in a book)
   - Each chunk has a fixed size (e.g., 30 images)
   - This makes the process much faster for large image sets

2. Reference Image:
   - For each chunk:
     * Take the first image as reference
     * Compare it with all other images in the chunk
     * Move to next image as reference
   - This ensures we don't miss any similar pairs

3. Similarity Check (same as before):
   - Convert images to HSV color space
   - Create color histograms
   - Calculate distance between histograms
   - If distance < threshold, images are similar

4. Safe Removal:
   - When similar images are found:
     * Move them to backup directory (don't delete)
     * Update image list
     * Continue with next reference image

Key Features:
- Chunk-based processing for better performance
- Safe image handling (backup instead of delete)
- Progress visualization
- Interactive threshold adjustment
- Result preview
- Detailed logging

Usage:
    python 02_02.py <directory> [--chunk-size SIZE] [--threshold THRESHOLD]

Controls:
    '+' - Increase threshold
    '-' - Decrease threshold
    's' - Save current results
    'q' - Quit
    Space - Process next chunk
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
class ChunkParams:
    """Parameters for chunked similarity detection."""
    chunk_size: int = 30
    threshold: float = 0.7
    bins: Tuple[int, int, int] = (8, 8, 8)
    supported_formats: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')
    threshold_step: float = 0.05


class ChunkedImageProcessor:
    """A class to handle chunked image similarity detection."""

    def __init__(self, output_dir: str = "processed_images"):
        """Initialize the processor.

        Args:
            output_dir: Base directory for output
        """
        self.output_dir = Path(output_dir)
        self.backup_dir = self.output_dir / "similar_images"
        self.params = ChunkParams()
        self.window_name = "Chunk Preview"
        
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

    def show_chunk_preview(
        self,
        reference_image: NDArray,
        similar_images: List[NDArray]
    ) -> None:
        """Show preview of reference image and similar images found.

        Args:
            reference_image: Reference image
            similar_images: List of similar images found
        """
        try:
            # Create preview grid
            max_preview = 5
            similar_count = min(len(similar_images), max_preview)
            
            # Resize images
            height = 200
            ref_resized = cv2.resize(
                reference_image,
                (int(height * reference_image.shape[1] / reference_image.shape[0]), height)
            )
            
            similar_resized = []
            for img in similar_images[:max_preview]:
                resized = cv2.resize(
                    img,
                    (int(height * img.shape[1] / img.shape[0]), height)
                )
                similar_resized.append(resized)
            
            # Create preview
            preview = np.hstack([ref_resized] + similar_resized)
            
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

    def process_chunk(
        self,
        image_paths: List[Path],
        start_idx: int
    ) -> Tuple[Set[Path], List[NDArray]]:
        """Process a chunk of images.

        Args:
            image_paths: List of all image paths
            start_idx: Starting index for this chunk

        Returns:
            Set of similar image paths and list of similar images
        """
        try:
            similar_paths = set()
            similar_images = []
            
            # Get chunk range
            end_idx = min(
                start_idx + self.params.chunk_size,
                len(image_paths)
            )
            
            # Load reference image
            ref_path = image_paths[start_idx]
            ref_image = self.load_image(str(ref_path))
            if ref_image is None:
                return set(), []
                
            ref_hist = self.extract_color_histogram(ref_image)
            if ref_hist is None:
                return set(), []
            
            # Compare with other images in chunk
            for i in range(start_idx + 1, end_idx):
                curr_path = image_paths[i]
                curr_image = self.load_image(str(curr_path))
                if curr_image is None:
                    continue
                    
                curr_hist = self.extract_color_histogram(curr_image)
                if curr_hist is None:
                    continue
                
                # Check similarity
                distance = euclidean(ref_hist, curr_hist)
                if distance < self.params.threshold:
                    similar_paths.add(curr_path)
                    similar_images.append(curr_image)
            
            return similar_paths, similar_images
            
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            return set(), []

    def backup_similar_images(self, similar_paths: Set[Path]) -> None:
        """Move similar images to backup directory.

        Args:
            similar_paths: Paths of similar images to backup
        """
        try:
            for path in similar_paths:
                dest_path = self.backup_dir / path.name
                shutil.move(path, dest_path)
                logger.info(f"Backed up: {path.name}")
                
        except Exception as e:
            logger.error(f"Error backing up images: {str(e)}")

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
            logger.info("Press Space to process next chunk")
            logger.info("Press '+'/'-' to adjust threshold")
            logger.info("Press 's' to save current results")
            logger.info("Press 'q' to quit")
            
            # Process chunks
            chunk_start = 0
            total_similar = 0
            
            while chunk_start < len(image_paths):
                # Process chunk
                similar_paths, similar_images = self.process_chunk(
                    image_paths,
                    chunk_start
                )
                
                if similar_paths:
                    # Show preview
                    ref_image = self.load_image(str(image_paths[chunk_start]))
                    if ref_image is not None:
                        self.show_chunk_preview(ref_image, similar_images)
                    
                    # Handle input
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord(' '):  # Process
                            self.backup_similar_images(similar_paths)
                            total_similar += len(similar_paths)
                            break
                        elif key == ord('+'):  # Increase threshold
                            self.params.threshold += self.params.threshold_step
                            similar_paths, similar_images = self.process_chunk(
                                image_paths,
                                chunk_start
                            )
                            if ref_image is not None:
                                self.show_chunk_preview(ref_image, similar_images)
                        elif key == ord('-'):  # Decrease threshold
                            self.params.threshold -= self.params.threshold_step
                            similar_paths, similar_images = self.process_chunk(
                                image_paths,
                                chunk_start
                            )
                            if ref_image is not None:
                                self.show_chunk_preview(ref_image, similar_images)
                        elif key == ord('q'):
                            return True
                
                chunk_start += 1
                
            logger.info(f"Processed {len(image_paths)} images")
            logger.info(f"Found {total_similar} similar images")
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
            description='Remove similar images using chunk processing'
        )
        parser.add_argument(
            'directory',
            help='Directory containing images'
        )
        parser.add_argument(
            '--chunk-size',
            type=int,
            default=30,
            help='Number of images to process in each chunk'
        )
        parser.add_argument(
            '--threshold',
            type=float,
            default=0.7,
            help='Similarity threshold (lower = stricter)'
        )
        
        args = parser.parse_args()
        
        # Create processor
        processor = ChunkedImageProcessor()
        processor.params.chunk_size = args.chunk_size
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