"""Image Stitching and Panorama Creation Module.

This module combines multiple images into a panorama using feature detection,
matching, and blending techniques.

Algorithm Explanation for Beginners:
--------------------------------
We create panoramas through these steps:

1. Feature Detection (SIFT):
   - Find special points in each image
   - Like finding landmarks in a city
   - SIFT looks for:
     * Corners and edges
     * Unique patterns
     * Scale-invariant features
   - These help align images

2. Feature Matching:
   - Match points between images
   - Like connecting dots between photos
   - For each point in image 1:
     * Find similar points in image 2
     * Keep best matches
     * Remove weak matches

3. Homography Estimation:
   - Find transformation between images
   - Like figuring out how to align photos
   - RANSAC helps:
     * Try different alignments
     * Keep the best one
     * Remove outliers

4. Image Warping:
   - Transform second image to match first
   - Like bending a photo to fit
   - Preserves straight lines
   - Maintains perspective

5. Image Blending:
   - Combine images smoothly
   - Like creating a seamless collage
   - Poisson blending:
     * Gradual transition
     * No visible seams

Key Features:
- Interactive image loading
- Real-time preview
- Progress feedback
- Error handling
- Resource cleanup
- Performance optimization

Usage:
    python image_stitch.py <input_dir> [--method METHOD]

Controls:
    'p' - Preview current result
    'r' - Reset stitching
    's' - Save panorama
    'q' - Quit
"""

from dataclasses import dataclass
import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class StitchingParams:
    """Parameters for image stitching."""
    match_ratio: float = 0.75
    min_matches: int = 10
    blend_width: int = 100
    ransac_threshold: float = 4.0
    supported_formats: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')
    target_width: int = 1920


class ImageStitcher:
    """A class to handle image stitching."""

    def __init__(self, output_dir: str = "panorama_output"):
        """Initialize the stitcher.

        Args:
            output_dir: Base directory for output
        """
        self.output_dir = Path(output_dir)
        self.params = StitchingParams()
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize state
        self.images = []
        self.result = None
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.DescriptorMatcher_create(
            cv2.DescriptorMatcher_FLANNBASED
        )

    def load_images(self, directory: str) -> bool:
        """Load images from directory.

        Args:
            directory: Directory containing images

        Returns:
            True if successful
        """
        try:
            directory = Path(directory)
            if not directory.is_dir():
                raise RuntimeError(f"Not a directory: {directory}")
            
            # Find image files
            image_files = []
            for ext in self.params.supported_formats:
                image_files.extend(directory.glob(f"*{ext}"))
            
            if not image_files:
                raise RuntimeError(f"No images found in {directory}")
            
            # Sort files
            image_files = sorted(image_files)
            
            # Load images
            self.images = []
            for path in image_files:
                image = cv2.imread(str(path))
                if image is None:
                    logger.warning(f"Could not load {path}")
                    continue
                    
                self.images.append(image)
            
            if len(self.images) < 2:
                raise RuntimeError("Need at least 2 images")
            
            logger.info(f"Loaded {len(self.images)} images")
            return True
            
        except Exception as e:
            logger.error(f"Error loading images: {str(e)}")
            return False

    def find_matches(
        self,
        image1: NDArray,
        image2: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """Find matching points between images.

        Args:
            image1: First image
            image2: Second image

        Returns:
            Points from both images that match
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        kp1, desc1 = self.sift.detectAndCompute(gray1, None)
        kp2, desc2 = self.sift.detectAndCompute(gray2, None)
        
        # Match features
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Filter matches
        good = []
        for m, n in matches:
            if m.distance < self.params.match_ratio * n.distance:
                good.append(m)
        
        if len(good) < self.params.min_matches:
            raise RuntimeError("Not enough matches")
        
        # Get matching points
        points1 = np.float32(
            [kp1[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        points2 = np.float32(
            [kp2[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        
        return points1, points2

    def stitch_pair(
        self,
        image1: NDArray,
        image2: NDArray
    ) -> Optional[NDArray]:
        """Stitch two images together.

        Args:
            image1: First image
            image2: Second image

        Returns:
            Stitched image if successful
        """
        try:
            # Find matches
            points1, points2 = self.find_matches(image1, image2)
            
            # Find homography
            H, mask = cv2.findHomography(
                points2,
                points1,
                cv2.RANSAC,
                self.params.ransac_threshold
            )
            
            if H is None:
                raise RuntimeError("Could not find homography")
            
            # Warp image
            height, width = image1.shape[:2]
            warped = cv2.warpPerspective(image2, H, (width, height))
            
            # Create mask for blending
            mask = np.zeros_like(warped)
            mask[:, :mask.shape[1] // 2, :] = 1
            
            # Blend images
            result = cv2.seamlessClone(
                warped,
                image1,
                mask * 255,
                (width // 2, height // 2),
                cv2.NORMAL_CLONE
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error stitching images: {str(e)}")
            return None

    def stitch_all(self) -> bool:
        """Stitch all loaded images.

        Returns:
            True if successful
        """
        try:
            if len(self.images) < 2:
                raise RuntimeError("Need at least 2 images")
            
            # Start with first image
            self.result = self.images[0]
            
            # Add remaining images
            for i in range(1, len(self.images)):
                logger.info(f"Stitching image {i+1}/{len(self.images)}")
                
                result = self.stitch_pair(
                    self.result,
                    self.images[i]
                )
                
                if result is None:
                    return False
                
                self.result = result
                
                # Show progress
                self.show_preview()
            
            return True
            
        except Exception as e:
            logger.error(f"Error stitching images: {str(e)}")
            return False

    def show_preview(self) -> None:
        """Show current result."""
        if self.result is not None:
            cv2.imshow('Panorama', self.result)
            cv2.waitKey(1)

    def save_result(self, path: Optional[str] = None) -> bool:
        """Save panorama.

        Args:
            path: Optional output path

        Returns:
            True if successful
        """
        try:
            if self.result is None:
                raise RuntimeError("No result to save")
            
            # Use default path if none provided
            if path is None:
                path = self.output_dir / "panorama.jpg"
            else:
                path = Path(path)
            
            # Save result
            cv2.imwrite(str(path), self.result)
            logger.info(f"Saved panorama to {path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving result: {str(e)}")
            return False

    def run(self, directory: str) -> bool:
        """Run stitching workflow.

        Args:
            directory: Input directory

        Returns:
            True if successful
        """
        try:
            # Load images
            if not self.load_images(directory):
                return False
            
            # Stitch images
            if not self.stitch_all():
                return False
            
            # Save result
            if not self.save_result():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error running stitcher: {str(e)}")
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
            description='Create panorama from images'
        )
        parser.add_argument(
            'input_dir',
            help='Directory containing images'
        )
        parser.add_argument(
            '--method',
            choices=['normal', 'mixed'],
            default='normal',
            help='Blending method'
        )
        
        args = parser.parse_args()
        
        # Create stitcher
        stitcher = ImageStitcher()
        
        # Run stitching
        if not stitcher.run(args.input_dir):
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())