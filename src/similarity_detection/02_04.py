"""Advanced Image Similarity Detection Module.

This module demonstrates and compares different image similarity detection methods,
from basic pixel comparison to advanced feature matching.

Algorithm Explanation for Beginners:
--------------------------------
We implement four different ways to compare images:

1. Feature Matching (ORB):
   - Find special points (features) in each image
   - These are like unique landmarks in the image
   - Match these points between images
   - More matching points = more similar images
   - Good for finding similar objects, even if rotated/scaled

2. Structural Similarity (SSIM):
   - Compare patterns of brightness and contrast
   - Works like human eyes - focuses on visible differences
   - Score from -1 to 1 (1 = identical images)
   - Good for finding overall image similarity

3. Mean Squared Error (MSE):
   - Compare images pixel by pixel
   - Calculate average difference in pixel values
   - Lower score = more similar images
   - Simple but sensitive to small changes

4. Color Histogram Comparison:
   - Create color "fingerprint" of each image
   - Compare these fingerprints
   - Fast but ignores image structure
   - Good for finding overall color similarity

Key Features:
- Multiple comparison methods
- Interactive visualization
- Method comparison view
- Progress feedback
- Detailed logging
- Safe image handling

Usage:
    python 02_04.py <image1> <image2> [--method METHOD]

Methods:
    'orb' - Feature matching with ORB
    'ssim' - Structural similarity
    'mse' - Mean squared error
    'hist' - Color histogram
    'all' - Compare all methods

Controls:
    'n' - Switch to next method
    'p' - Switch to previous method
    'q' - Quit
"""

from dataclasses import dataclass
import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import euclidean
from skimage.metrics import structural_similarity as ssim


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonParams:
    """Parameters for image comparison."""
    orb_features: int = 500
    hist_bins: Tuple[int, int, int] = (8, 8, 8)
    supported_formats: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')


class ImageComparator:
    """A class to handle multiple image comparison methods."""

    def __init__(self):
        """Initialize the comparator."""
        self.params = ComparisonParams()
        self.window_name = "Comparison Result"
        self.current_method = 'all'
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(
            nfeatures=self.params.orb_features
        )
        
        # Initialize feature matcher
        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=True
        )

    def load_image(
        self,
        path: str,
        grayscale: bool = False
    ) -> Optional[NDArray]:
        """Load an image from file.

        Args:
            path: Path to image file
            grayscale: Whether to load in grayscale

        Returns:
            Loaded image
        """
        try:
            # Load image
            if grayscale:
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(path)
                
            if image is None:
                raise RuntimeError(f"Could not read image: {path}")
                
            return image
            
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None

    def compare_orb(
        self,
        image1: NDArray,
        image2: NDArray
    ) -> Tuple[float, Optional[NDArray]]:
        """Compare images using ORB features.

        Args:
            image1: First image
            image2: Second image

        Returns:
            Similarity score and visualization
        """
        try:
            # Convert to grayscale if needed
            if len(image1.shape) == 3:
                gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            else:
                gray1, gray2 = image1, image2
            
            # Detect and compute
            kp1, des1 = self.orb.detectAndCompute(gray1, None)
            kp2, des2 = self.orb.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None:
                return 0.0, None
            
            # Match features
            matches = self.matcher.match(des1, des2)
            
            # Calculate score
            min_dist = float('inf')
            max_dist = 0
            
            for m in matches:
                if m.distance < min_dist:
                    min_dist = m.distance
                if m.distance > max_dist:
                    max_dist = m.distance
            
            if max_dist == 0:
                return 0.0, None
                
            # Normalize score
            score = 1.0 - (sum(m.distance for m in matches) / (len(matches) * max_dist))
            
            # Create visualization
            matches = sorted(matches, key=lambda x: x.distance)
            vis = cv2.drawMatches(
                image1, kp1,
                image2, kp2,
                matches[:10], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            return score, vis
            
        except Exception as e:
            logger.error(f"Error in ORB comparison: {str(e)}")
            return 0.0, None

    def compare_ssim(
        self,
        image1: NDArray,
        image2: NDArray
    ) -> Tuple[float, Optional[NDArray]]:
        """Compare images using SSIM.

        Args:
            image1: First image
            image2: Second image

        Returns:
            Similarity score and visualization
        """
        try:
            # Convert to grayscale if needed
            if len(image1.shape) == 3:
                gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            else:
                gray1, gray2 = image1, image2
            
            # Calculate SSIM
            score, diff = ssim(gray1, gray2, full=True)
            
            # Create visualization
            diff = (diff * 255).astype(np.uint8)
            vis = np.hstack([gray1, gray2, diff])
            
            return score, vis
            
        except Exception as e:
            logger.error(f"Error in SSIM comparison: {str(e)}")
            return 0.0, None

    def compare_mse(
        self,
        image1: NDArray,
        image2: NDArray
    ) -> Tuple[float, Optional[NDArray]]:
        """Compare images using MSE.

        Args:
            image1: First image
            image2: Second image

        Returns:
            Similarity score and visualization
        """
        try:
            # Calculate MSE
            err = np.sum((image1.astype(float) - image2.astype(float)) ** 2)
            err /= float(image1.shape[0] * image1.shape[1])
            
            # Normalize to 0-1 range (inverted)
            max_mse = 255.0 ** 2  # Maximum possible MSE
            score = 1.0 - (err / max_mse)
            
            # Create visualization
            diff = cv2.absdiff(image1, image2)
            vis = np.hstack([image1, image2, diff])
            
            return score, vis
            
        except Exception as e:
            logger.error(f"Error in MSE comparison: {str(e)}")
            return 0.0, None

    def compare_hist(
        self,
        image1: NDArray,
        image2: NDArray
    ) -> Tuple[float, Optional[NDArray]]:
        """Compare images using color histograms.

        Args:
            image1: First image
            image2: Second image

        Returns:
            Similarity score and visualization
        """
        try:
            # Convert to HSV
            hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms
            hist1 = cv2.calcHist(
                [hsv1], [0, 1, 2], None,
                self.params.hist_bins,
                [0, 256, 0, 256, 0, 256]
            )
            hist2 = cv2.calcHist(
                [hsv2], [0, 1, 2], None,
                self.params.hist_bins,
                [0, 256, 0, 256, 0, 256]
            )
            
            # Normalize
            cv2.normalize(hist1, hist1)
            cv2.normalize(hist2, hist2)
            
            # Calculate similarity
            score = 1.0 - euclidean(
                hist1.flatten(),
                hist2.flatten()
            ) / np.sqrt(2.0)
            
            # Create visualization
            vis = np.hstack([image1, image2])
            
            return score, vis
            
        except Exception as e:
            logger.error(f"Error in histogram comparison: {str(e)}")
            return 0.0, None

    def show_comparison(
        self,
        method: str,
        score: float,
        vis: Optional[NDArray]
    ) -> None:
        """Show comparison visualization.

        Args:
            method: Comparison method used
            score: Similarity score
            vis: Visualization image
        """
        try:
            if vis is None:
                return
                
            # Add text
            height = 50
            text = f"{method.upper()}: {score:.3f}"
            
            header = np.zeros((height, vis.shape[1], 3), dtype=np.uint8)
            cv2.putText(
                header,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Stack header and visualization
            if len(vis.shape) == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            
            display = np.vstack([header, vis])
            
            # Show result
            cv2.imshow(self.window_name, display)
            
        except Exception as e:
            logger.error(f"Error showing comparison: {str(e)}")

    def compare_images(
        self,
        image1_path: str,
        image2_path: str,
        method: str = 'all'
    ) -> bool:
        """Compare two images using specified method.

        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            method: Comparison method to use

        Returns:
            True if successful
        """
        try:
            # Load images
            image1 = self.load_image(image1_path)
            image2 = self.load_image(image2_path)
            
            if image1 is None or image2 is None:
                return False
            
            # Resize if different sizes
            if image1.shape != image2.shape:
                height = min(image1.shape[0], image2.shape[0])
                width = min(image1.shape[1], image2.shape[1])
                
                image1 = cv2.resize(image1, (width, height))
                image2 = cv2.resize(image2, (width, height))
            
            methods = {
                'orb': self.compare_orb,
                'ssim': self.compare_ssim,
                'mse': self.compare_mse,
                'hist': self.compare_hist
            }
            
            if method == 'all':
                # Show all methods
                logger.info("\nComparison Results:")
                
                for name, func in methods.items():
                    score, vis = func(image1, image2)
                    logger.info(f"{name.upper()}: {score:.3f}")
                    self.show_comparison(name, score, vis)
                    
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        break
                        
            elif method in methods:
                # Show single method
                score, vis = methods[method](image1, image2)
                logger.info(f"\n{method.upper()}: {score:.3f}")
                self.show_comparison(method, score, vis)
                cv2.waitKey(0)
                
            else:
                logger.error(f"Unknown method: {method}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error comparing images: {str(e)}")
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
            description='Compare images using various methods'
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
            '--method',
            choices=['orb', 'ssim', 'mse', 'hist', 'all'],
            default='all',
            help='Comparison method to use'
        )
        
        args = parser.parse_args()
        
        # Create comparator
        comparator = ImageComparator()
        
        # Compare images
        if comparator.compare_images(
            args.image1,
            args.image2,
            args.method
        ):
            logger.info("Comparison completed successfully")
            return 0
        else:
            logger.error("Comparison failed")
            return 1

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
