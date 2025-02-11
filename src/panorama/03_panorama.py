"""Panorama Image Creation Module.

This module demonstrates panorama creation by stitching multiple images together using
various feature detection methods and homography-based alignment.

Algorithm Explanation for Beginners:
--------------------------------
Creating a panorama involves several steps to combine multiple images:

1. Feature Detection:
   - We find special points (features) in each image
   - We can use different detectors:
     * BRISK: Fast but less accurate
     * SIFT: Slower but very accurate
     * AKAZE: Good balance of speed and accuracy
     * ORB: Fast and free to use
   - Each feature has a unique "fingerprint" (descriptor)
   
2. Feature Matching:
   - For each feature in one image:
     * Find the most similar feature in the other image
     * Filter matches based on quality
   - This tells us which parts of the images overlap
   
3. Image Alignment (Homography):
   - Using the matched features:
     * Calculate how to transform one image to align with the other
     * This creates a mathematical formula (matrix)
     * RANSAC helps remove incorrect matches
   
4. Image Stitching:
   - Transform one image using the calculated formula
   - Blend the images together where they overlap
   - Result is a seamless panorama

The process shows:
- Detected features in each image
- Matches between images
- Final stitched panorama
- Progress feedback at each step

Key Features:
- Multiple feature detector options
- Interactive visualization
- Progress feedback
- Result saving
- Error handling

Usage:
    python 03_panorama.py <image1_path> <image2_path> <output_path> [--detector TYPE]

Controls:
    Any key - proceed to next step
    'q' - quit
    's' - save current view
    'd' - cycle through detectors
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Literal
import argparse

import cv2
import numpy as np
from numpy.typing import NDArray


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


# Type aliases
FeatureDetectorType = Literal['BRISK', 'SIFT', 'AKAZE', 'ORB']
KeyPoints = List[cv2.KeyPoint]
Matches = List[cv2.DMatch]


@dataclass
class PanoramaParams:
    """Parameters for panorama creation."""
    detector_type: FeatureDetectorType = 'BRISK'
    ransac_threshold: float = 5.0
    min_matches: int = 10
    cross_check: bool = True
    
    def cycle_detector(self) -> None:
        """Cycle through available detectors."""
        detectors = ['BRISK', 'SIFT', 'AKAZE', 'ORB']
        current_idx = detectors.index(self.detector_type)
        self.detector_type = detectors[(current_idx + 1) % len(detectors)]


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
        self.detectors: Dict[str, cv2.Feature2D] = {}
        
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

    def create_feature_detector(self) -> Optional[cv2.Feature2D]:
        """Create feature detector of current type.

        Returns:
            Feature detector object
        """
        try:
            if self.params.detector_type not in self.detectors:
                detectors = {
                    'BRISK': cv2.BRISK_create,
                    'SIFT': cv2.SIFT_create,
                    'AKAZE': cv2.AKAZE_create,
                    'ORB': cv2.ORB_create
                }
                self.detectors[self.params.detector_type] = detectors[
                    self.params.detector_type
                ]()
                
            return self.detectors[self.params.detector_type]
            
        except Exception as e:
            logger.error(f"Error creating detector: {str(e)}")
            return None

    def detect_features(
        self,
        img1: NDArray,
        img2: NDArray
    ) -> Tuple[Optional[KeyPoints], Optional[KeyPoints], Optional[NDArray], Optional[NDArray]]:
        """Detect features in both images.

        Args:
            img1: First image
            img2: Second image

        Returns:
            Tuple of (keypoints1, keypoints2, descriptors1, descriptors2)
        """
        try:
            detector = self.create_feature_detector()
            if detector is None:
                return None, None, None, None
                
            # Detect and compute
            keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
            keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
            
            return keypoints1, keypoints2, descriptors1, descriptors2
            
        except Exception as e:
            logger.error(f"Error detecting features: {str(e)}")
            return None, None, None, None

    def match_features(
        self,
        descriptors1: NDArray,
        descriptors2: NDArray
    ) -> Optional[Matches]:
        """Match features between images.

        Args:
            descriptors1: First image descriptors
            descriptors2: Second image descriptors

        Returns:
            List of good matches
        """
        try:
            # Create matcher based on detector type
            if self.params.detector_type == 'SIFT':
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=self.params.cross_check)
            else:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=self.params.cross_check)
            
            # Match features
            matches = matcher.match(descriptors1, descriptors2)
            
            # Sort by distance
            return sorted(matches, key=lambda x: x.distance)
            
        except Exception as e:
            logger.error(f"Error matching features: {str(e)}")
            return None

    def find_homography(
        self,
        keypoints1: KeyPoints,
        keypoints2: KeyPoints,
        matches: Matches
    ) -> Optional[NDArray]:
        """Find homography matrix.

        Args:
            keypoints1: First image keypoints
            keypoints2: Second image keypoints
            matches: Feature matches

        Returns:
            Homography matrix
        """
        try:
            if len(matches) < self.params.min_matches:
                logger.error(
                    f"Not enough matches: {len(matches)}/{self.params.min_matches}"
                )
                return None
                
            # Extract matched points
            src_pts = np.float32(
                [keypoints1[m.queryIdx].pt for m in matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [keypoints2[m.trainIdx].pt for m in matches]
            ).reshape(-1, 1, 2)
            
            # Find homography
            H, _ = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                self.params.ransac_threshold
            )
            
            return H
            
        except Exception as e:
            logger.error(f"Error finding homography: {str(e)}")
            return None

    def create_panorama(
        self,
        img1: NDArray,
        img2: NDArray,
        H: NDArray
    ) -> Optional[NDArray]:
        """Create panorama by warping and stitching.

        Args:
            img1: First image
            img2: Second image
            H: Homography matrix

        Returns:
            Stitched panorama
        """
        try:
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
            
        except Exception as e:
            logger.error(f"Error creating panorama: {str(e)}")
            return None

    def draw_features(
        self,
        img1: NDArray,
        img2: NDArray,
        keypoints1: KeyPoints,
        keypoints2: KeyPoints
    ) -> Optional[NDArray]:
        """Draw detected features.

        Args:
            img1: First image
            img2: Second image
            keypoints1: First image keypoints
            keypoints2: Second image keypoints

        Returns:
            Combined image with features
        """
        try:
            # Draw keypoints
            img1_kp = cv2.drawKeypoints(
                img1,
                keypoints1,
                None,
                color=(0, 255, 0),
                flags=0
            )
            img2_kp = cv2.drawKeypoints(
                img2,
                keypoints2,
                None,
                color=(0, 255, 0),
                flags=0
            )
            
            # Combine images
            return cv2.hconcat([img1_kp, img2_kp])
            
        except Exception as e:
            logger.error(f"Error drawing features: {str(e)}")
            return None

    def draw_matches(
        self,
        img1: NDArray,
        img2: NDArray,
        keypoints1: KeyPoints,
        keypoints2: KeyPoints,
        matches: Matches,
        max_matches: int = 25
    ) -> Optional[NDArray]:
        """Draw feature matches.

        Args:
            img1: First image
            img2: Second image
            keypoints1: First image keypoints
            keypoints2: Second image keypoints
            matches: Feature matches
            max_matches: Maximum matches to draw

        Returns:
            Image with matches drawn
        """
        try:
            return cv2.drawMatches(
                img1,
                keypoints1,
                img2,
                keypoints2,
                matches[:max_matches],
                None,
                flags=2
            )
        except Exception as e:
            logger.error(f"Error drawing matches: {str(e)}")
            return None

    def save_image(self, image: NDArray, name: str) -> None:
        """Save image to file.

        Args:
            image: Image to save
            name: Base name for the file
        """
        filename = self.output_dir / f"{name}.jpg"
        cv2.imwrite(str(filename), image)
        logger.info(f"Saved image to: {filename}")

    def create(
        self,
        image1_path: str,
        image2_path: str,
        output_path: str
    ) -> bool:
        """Create panorama from two images.

        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            output_path: Path to save final panorama

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
            logger.info("Press 'd' to cycle through detectors")

            while True:
                # Detect features
                keypoints1, keypoints2, descriptors1, descriptors2 = self.detect_features(
                    img1,
                    img2
                )
                if None in (keypoints1, keypoints2, descriptors1, descriptors2):
                    return False
                    
                logger.info(
                    f"Using {self.params.detector_type} detector - "
                    f"Found features: {len(keypoints1)}, {len(keypoints2)}"
                )
                
                # Show features
                features_image = self.draw_features(
                    img1,
                    img2,
                    keypoints1,
                    keypoints2
                )
                if features_image is None:
                    return False
                    
                cv2.imshow(self.window_name, features_image)
                
                # Handle input
                key = cv2.waitKey(0)
                if key == ord('q'):
                    return True
                elif key == ord('s'):
                    self.save_image(features_image, "features")
                elif key == ord('d'):
                    self.params.cycle_detector()
                    continue
                
                # Match features
                matches = self.match_features(descriptors1, descriptors2)
                if matches is None:
                    return False
                    
                logger.info(f"Found {len(matches)} matches")
                
                # Show matches
                matches_image = self.draw_matches(
                    img1,
                    img2,
                    keypoints1,
                    keypoints2,
                    matches
                )
                if matches_image is None:
                    return False
                    
                cv2.imshow(self.window_name, matches_image)
                
                # Handle input
                key = cv2.waitKey(0)
                if key == ord('q'):
                    return True
                elif key == ord('s'):
                    self.save_image(matches_image, "matches")
                elif key == ord('d'):
                    self.params.cycle_detector()
                    continue

                # Find homography
                H = self.find_homography(keypoints1, keypoints2, matches)
                if H is None:
                    return False
                    
                # Create panorama
                panorama = self.create_panorama(img1, img2, H)
                if panorama is None:
                    return False
                    
                cv2.imshow(self.window_name, panorama)
                
                # Handle input
                key = cv2.waitKey(0)
                if key == ord('s'):
                    cv2.imwrite(output_path, panorama)
                    logger.info(f"Saved panorama to: {output_path}")
                elif key == ord('d'):
                    self.params.cycle_detector()
                    continue
                    
                break

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
        # Parse arguments
        parser = argparse.ArgumentParser(
            description='Create panorama from two images'
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
            'output',
            help='Path to save panorama'
        )
        parser.add_argument(
            '--detector',
            choices=['BRISK', 'SIFT', 'AKAZE', 'ORB'],
            default='BRISK',
            help='Feature detector type'
        )
        
        args = parser.parse_args()
        
        # Create panorama
        creator = PanoramaCreator()
        creator.params.detector_type = args.detector
        
        if creator.create(args.image1, args.image2, args.output):
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