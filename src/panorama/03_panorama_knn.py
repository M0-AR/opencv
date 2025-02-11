"""Panorama Creation with KNN Feature Matching Module.

This module demonstrates panorama creation by stitching two images together using
BRISK feature detection and k-Nearest Neighbors (KNN) matching.

Algorithm Explanation for Beginners:
--------------------------------
Creating a panorama involves several steps to combine two images:

1. Feature Detection (BRISK):
   - We find special points (features) in both images
   - These are unique points that we can easily recognize
   - BRISK looks for corners and distinctive patterns
   - Each feature has a unique "fingerprint" (descriptor)
   
2. Feature Matching (KNN):
   - For each feature in the first image:
     * Find the two most similar features in the second image
     * If the best match is much better than the second best
     * Keep it as a "good match"
   - This helps avoid incorrect matches
   
3. Image Alignment (Homography):
   - Using the good matches, we figure out how to align the images
   - We calculate a transformation matrix (homography)
   - This tells us how to warp the second image
   
4. Image Stitching:
   - We transform the second image using the homography
   - Then combine it with the first image
   - The result is a seamless panorama

The result shows:
- Feature matches between images
- Final stitched panorama
- Images are saved to disk for review

This helps us understand how computers can automatically combine photos into panoramas.

Key Features:
- BRISK feature detection
- KNN feature matching
- RANSAC homography estimation
- Interactive visualization
- Result saving

Usage:
    python 03_panorama_knn.py <image1_path> <image2_path>

Controls:
    Any key - proceed to next step
    'q' - quit
    's' - save current view
"""

import logging
import os
from pathlib import Path
import sys
from typing import List, Optional, Tuple
from dataclasses import dataclass

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
class PanoramaParams:
    """Parameters for panorama creation."""
    ratio_threshold: float = 0.8  # Lowe's ratio test threshold
    min_matches: int = 10  # Minimum matches required
    ransac_threshold: float = 5.0  # RANSAC threshold
    
    def reset(self) -> None:
        """Reset parameters to default values."""
        self.__init__()


class PanoramaCreator:
    """A class to handle panorama creation and visualization."""

    def __init__(self, output_dir: str = "panorama_output"):
        """Initialize the creator.

        Args:
            output_dir: Directory to save output images
        """
        self.output_dir = Path(output_dir)
        self.window_name = 'Panorama Creation'
        self.params = PanoramaParams()
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

    def load_images(
        self,
        image1_path: str,
        image2_path: str
    ) -> Tuple[Optional[NDArray], Optional[NDArray]]:
        """Load input images.

        Args:
            image1_path: Path to first image
            image2_path: Path to second image

        Returns:
            Tuple of (first image, second image)
        """
        try:
            if not os.path.exists(image1_path):
                raise FileNotFoundError(f"First image not found: {image1_path}")
            if not os.path.exists(image2_path):
                raise FileNotFoundError(f"Second image not found: {image2_path}")
                
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)
            
            if img1 is None:
                raise RuntimeError(f"Could not read first image: {image1_path}")
            if img2 is None:
                raise RuntimeError(f"Could not read second image: {image2_path}")
                
            return img1, img2
            
        except Exception as e:
            logger.error(f"Error loading images: {str(e)}")
            return None, None

    def detect_features(
        self,
        img1: NDArray,
        img2: NDArray
    ) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], NDArray, NDArray]:
        """Detect BRISK features in images.

        Args:
            img1: First image
            img2: Second image

        Returns:
            Tuple of (keypoints1, keypoints2, descriptors1, descriptors2)
        """
        # Initialize BRISK detector
        brisk = cv2.BRISK_create()
        
        # Detect and compute
        keypoints1, descriptors1 = brisk.detectAndCompute(img1, None)
        keypoints2, descriptors2 = brisk.detectAndCompute(img2, None)
        
        return keypoints1, keypoints2, descriptors1, descriptors2

    def match_features(
        self,
        descriptors1: NDArray,
        descriptors2: NDArray
    ) -> List[cv2.DMatch]:
        """Match features using KNN.

        Args:
            descriptors1: First image descriptors
            descriptors2: Second image descriptors

        Returns:
            List of good matches
        """
        # Create BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Perform KNN matching
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.params.ratio_threshold * n.distance:
                good_matches.append(m)
                
        # Sort by distance
        return sorted(good_matches, key=lambda x: x.distance)

    def find_homography(
        self,
        keypoints1: List[cv2.KeyPoint],
        keypoints2: List[cv2.KeyPoint],
        good_matches: List[cv2.DMatch]
    ) -> Tuple[Optional[NDArray], Optional[NDArray]]:
        """Find homography matrix.

        Args:
            keypoints1: First image keypoints
            keypoints2: Second image keypoints
            good_matches: List of good matches

        Returns:
            Tuple of (homography matrix, mask)
        """
        if len(good_matches) < self.params.min_matches:
            logger.error(
                f"Not enough matches: {len(good_matches)}/{self.params.min_matches}"
            )
            return None, None
            
        # Extract matched point coordinates
        points1 = np.float32(
            [keypoints1[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        points2 = np.float32(
            [keypoints2[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(
            points1,
            points2,
            cv2.RANSAC,
            self.params.ransac_threshold
        )
        
        return H, mask

    def create_panorama(
        self,
        img1: NDArray,
        img2: NDArray,
        H: NDArray
    ) -> NDArray:
        """Create panorama by warping and combining images.

        Args:
            img1: First image
            img2: Second image
            H: Homography matrix

        Returns:
            Stitched panorama image
        """
        # Get dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Warp first image
        img1_transformed = cv2.warpPerspective(
            img1,
            H,
            (w2, h2)
        )
        
        # Combine images
        result = cv2.hconcat([img1_transformed, img2])
        
        return result

    def draw_matches(
        self,
        img1: NDArray,
        img2: NDArray,
        keypoints1: List[cv2.KeyPoint],
        keypoints2: List[cv2.KeyPoint],
        good_matches: List[cv2.DMatch],
        max_matches: int = 25
    ) -> NDArray:
        """Draw feature matches.

        Args:
            img1: First image
            img2: Second image
            keypoints1: First image keypoints
            keypoints2: Second image keypoints
            good_matches: List of good matches
            max_matches: Maximum matches to draw

        Returns:
            Image with matches drawn
        """
        return cv2.drawMatches(
            img1,
            keypoints1,
            img2,
            keypoints2,
            good_matches[:max_matches],
            None,
            flags=2
        )

    def save_image(self, image: NDArray, name: str) -> None:
        """Save image to file.

        Args:
            image: Image to save
            name: Base name for the file
        """
        filename = self.output_dir / f"{name}.jpg"
        cv2.imwrite(str(filename), image)
        logger.info(f"Saved image to: {filename}")

    def create(self, image1_path: str, image2_path: str) -> bool:
        """Create panorama from two images.

        Args:
            image1_path: Path to first image
            image2_path: Path to second image

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load images
            img1, img2 = self.load_images(image1_path, image2_path)
            if img1 is None or img2 is None:
                return False

            logger.info("Starting panorama creation...")
            logger.info("Press any key to proceed through steps")
            logger.info("Press 'q' to quit")
            logger.info("Press 's' to save current view")

            # Detect features
            keypoints1, keypoints2, descriptors1, descriptors2 = self.detect_features(
                img1,
                img2
            )
            logger.info(
                f"Detected features - Image 1: {len(keypoints1)}, "
                f"Image 2: {len(keypoints2)}"
            )

            # Match features
            good_matches = self.match_features(descriptors1, descriptors2)
            logger.info(f"Found {len(good_matches)} good matches")
            
            # Draw and show matches
            matches_image = self.draw_matches(
                img1,
                img2,
                keypoints1,
                keypoints2,
                good_matches
            )
            cv2.imshow(self.window_name, matches_image)
            
            # Wait for user input
            key = cv2.waitKey(0)
            if key == ord('q'):
                return True
            elif key == ord('s'):
                self.save_image(matches_image, "matches")

            # Find homography
            H, mask = self.find_homography(keypoints1, keypoints2, good_matches)
            if H is None:
                return False
                
            # Create panorama
            panorama = self.create_panorama(img1, img2, H)
            cv2.imshow(self.window_name, panorama)
            
            # Wait for user input
            key = cv2.waitKey(0)
            if key == ord('s'):
                self.save_image(panorama, "panorama")

            return True

        except Exception as e:
            logger.error(f"Error creating panorama: {str(e)}")
            return False

        finally:
            cv2.destroyAllWindows()


def main() -> int:
    """Main function.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Check command line arguments
        if len(sys.argv) != 3:
            logger.error("Usage: python 03_panorama_knn.py <image1_path> <image2_path>")
            return 1

        image1_path = sys.argv[1]
        image2_path = sys.argv[2]
        
        # Create panorama
        creator = PanoramaCreator()
        if creator.create(image1_path, image2_path):
            logger.info("Panorama creation completed successfully")
            return 0
        else:
            logger.error("Panorama creation failed")
            return 1

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
