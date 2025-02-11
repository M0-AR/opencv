"""Advanced Feature Extraction and Image Enhancement Module.

This module implements advanced feature extraction techniques combined with
various image enhancement methods. It includes:

Enhancement Methods:
- Multi-Scale Retinex (MSR)
- Gamma Correction
- Histogram Equalization
- Image Sharpening

Feature Extraction Methods:
- ORB (Oriented FAST and Rotated BRIEF)
- BRISK (Binary Robust Invariant Scalable Keypoints)
- Local Binary Patterns (LBP)
- Gray-Level Co-occurrence Matrix (GLCM)

The module processes video frames, enhances them using multiple techniques,
and extracts various features for analysis and comparison.

Date: 2025-02-11
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, NamedTuple
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EnhancementConfig:
    """Configuration for image enhancement."""
    msr_sigmas: List[int] = field(default_factory=lambda: [15, 80, 250])
    gamma: float = 1.5
    sharpen_kernel: NDArray = field(default_factory=lambda: np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]))


@dataclass
class TextureConfig:
    """Configuration for texture analysis."""
    lbp_points: int = 8
    lbp_radius: int = 1
    lbp_method: str = 'uniform'
    glcm_distances: List[int] = field(default_factory=lambda: [1])
    glcm_angles: List[float] = field(default_factory=lambda: [0])
    glcm_levels: int = 256


@dataclass
class ProcessingConfig:
    """Configuration for video processing."""
    crop_start: int = 200
    output_dir: str = "feature_extraction_output"
    save_format: str = "png"
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)
    texture: TextureConfig = field(default_factory=TextureConfig)
    detector_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        'fast_brief': (0, 0, 255),    # Red
        'fast': (0, 0, 255),          # Red (fallback)
        'orb': (255, 255, 0),         # Cyan
        'brisk': (0, 255, 255),       # Yellow
        'enhanced_orb': (255, 0, 0),  # Blue
        'enhanced_brisk': (0, 255, 0) # Green
    })


class TextureProperties(NamedTuple):
    """Properties extracted from texture analysis."""
    contrast: float
    dissimilarity: float
    homogeneity: float
    energy: float
    correlation: float


class ImageEnhancer:
    """Class for image enhancement operations."""

    def __init__(self, config: EnhancementConfig):
        """Initialize image enhancer.

        Args:
            config: Enhancement configuration parameters
        """
        self.config = config

    def multi_scale_retinex(self, image: NDArray) -> NDArray:
        """Apply Multi-Scale Retinex enhancement.

        Args:
            image: Input image

        Returns:
            Enhanced image using MSR
        """
        retinex = np.zeros_like(image, dtype=np.float32)
        for sigma in self.config.msr_sigmas:
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            retinex += np.log10(image + 1.0) - np.log10(blurred + 1.0)
        retinex /= len(self.config.msr_sigmas)
        return retinex

    def gamma_correction(self, image: NDArray) -> NDArray:
        """Apply gamma correction.

        Args:
            image: Input image

        Returns:
            Gamma corrected image
        """
        inv_gamma = 1.0 / self.config.gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]).astype("uint8")
        return cv2.LUT(image, table)

    def sharpen(self, image: NDArray) -> NDArray:
        """Apply sharpening filter.

        Args:
            image: Input image

        Returns:
            Sharpened image
        """
        return cv2.filter2D(image, -1, self.config.sharpen_kernel)

    def enhance_image(self, image: NDArray) -> Dict[str, NDArray]:
        """Apply all enhancement methods to an image.

        Args:
            image: Input image

        Returns:
            Dictionary of enhanced images
        """
        return {
            'msr': self.multi_scale_retinex(image),
            'gamma': self.gamma_correction(image),
            'equalized': cv2.equalizeHist(image),
            'sharpened': self.sharpen(image)
        }


class TextureAnalyzer:
    """Class for texture analysis operations."""

    def __init__(self, config: TextureConfig):
        """Initialize texture analyzer.

        Args:
            config: Texture analysis configuration
        """
        self.config = config

    def compute_lbp(self, image: NDArray) -> NDArray:
        """Compute Local Binary Pattern.

        Args:
            image: Input image

        Returns:
            LBP image
        """
        return local_binary_pattern(
            image,
            P=self.config.lbp_points,
            R=self.config.lbp_radius,
            method=self.config.lbp_method
        )

    def compute_glcm_properties(self, image: NDArray) -> TextureProperties:
        """Compute GLCM properties.

        Args:
            image: Input image

        Returns:
            TextureProperties containing GLCM measurements
        """
        glcm = graycomatrix(
            image,
            self.config.glcm_distances,
            self.config.glcm_angles,
            self.config.glcm_levels,
            symmetric=True,
            normed=True
        )

        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        values = [float(graycoprops(glcm, prop)[0, 0]) for prop in props]

        return TextureProperties(*values)


class FeatureExtractor:
    """Class for feature extraction operations."""

    def __init__(
        self,
        config: ProcessingConfig,
        enhancer: ImageEnhancer,
        analyzer: TextureAnalyzer
    ):
        """Initialize feature extractor.

        Args:
            config: Processing configuration
            enhancer: Image enhancer instance
            analyzer: Texture analyzer instance
        """
        self.config = config
        self.enhancer = enhancer
        self.analyzer = analyzer
        self._initialize_detectors()

    def _initialize_detectors(self) -> None:
        """Initialize feature detectors."""
        try:
            self.detectors = {
                'fast': cv2.FastFeatureDetector_create(),
                'orb': cv2.ORB_create(),
                'brisk': cv2.BRISK_create()
            }

            if hasattr(cv2.xfeatures2d, 'BriefDescriptorExtractor_create'):
                self.detectors['brief'] = cv2.xfeatures2d.BriefDescriptorExtractor_create()
                logger.info("BRIEF descriptor available")
            else:
                logger.warning("BRIEF descriptor not available")

        except Exception as e:
            logger.error(f"Error initializing detectors: {str(e)}")
            raise

    def save_enhanced_images(
        self,
        enhanced_images: Dict[str, NDArray],
        frame_number: int
    ) -> None:
        """Save enhanced images.

        Args:
            enhanced_images: Dictionary of enhanced images
            frame_number: Current frame number
        """
        for name, image in enhanced_images.items():
            filename = os.path.join(
                self.config.output_dir,
                f"enhanced_{name}_{frame_number:04d}.{self.config.save_format}"
            )
            cv2.imwrite(filename, image)

    def save_keypoints(
        self,
        image: NDArray,
        keypoints: List[cv2.KeyPoint],
        detector_name: str,
        frame_number: int
    ) -> None:
        """Save keypoint visualization.

        Args:
            image: Input image
            keypoints: Detected keypoints
            detector_name: Name of the detector
            frame_number: Current frame number
        """
        color = self.config.detector_colors.get(detector_name, (255, 255, 255))
        visualization = cv2.drawKeypoints(
            image,
            keypoints,
            None,
            color=color,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # Add keypoint count
        cv2.putText(
            visualization,
            f"Keypoints: {len(keypoints)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        filename = os.path.join(
            self.config.output_dir,
            f"keypoints_{detector_name}_{frame_number:04d}.{self.config.save_format}"
        )
        cv2.imwrite(filename, visualization)

    def save_texture_properties(
        self,
        props: TextureProperties,
        frame_number: int
    ) -> None:
        """Save texture properties.

        Args:
            props: Texture properties
            frame_number: Current frame number
        """
        filename = os.path.join(
            self.config.output_dir,
            f"texture_properties_{frame_number:04d}.json"
        )
        with open(filename, 'w') as f:
            json.dump(props._asdict(), f, indent=4)

    def process_frame(self, frame: NDArray, frame_number: int) -> None:
        """Process a single frame.

        Args:
            frame: Input frame
            frame_number: Current frame number
        """
        try:
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Enhance images
            enhanced_images = self.enhancer.enhance_image(gray_frame)
            self.save_enhanced_images(enhanced_images, frame_number)

            # Extract features from original and enhanced images
            for detector_name, detector in self.detectors.items():
                if detector_name == 'brief':
                    continue

                # Original image
                keypoints, _ = detector.detectAndCompute(gray_frame, None)
                self.save_keypoints(gray_frame, keypoints, detector_name, frame_number)

                # Enhanced image (MSR)
                enhanced_keypoints, _ = detector.detectAndCompute(enhanced_images['msr'], None)
                self.save_keypoints(
                    enhanced_images['msr'],
                    enhanced_keypoints,
                    f"enhanced_{detector_name}",
                    frame_number
                )

            # Compute and save texture properties
            lbp = self.analyzer.compute_lbp(gray_frame)
            cv2.imwrite(
                os.path.join(self.config.output_dir, f"lbp_{frame_number:04d}.{self.config.save_format}"),
                lbp
            )

            texture_props = self.analyzer.compute_glcm_properties(gray_frame)
            self.save_texture_properties(texture_props, frame_number)

        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {str(e)}")


def process_video(video_path: str, config: ProcessingConfig) -> bool:
    """Process video file with feature extraction.

    Args:
        video_path: Path to input video
        config: Processing configuration

    Returns:
        True if processing successful, False otherwise
    """
    try:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Initialize video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Processing video: {video_path.name} ({fps:.1f} fps, {total_frames} frames)")

        # Initialize components
        enhancer = ImageEnhancer(config.enhancement)
        analyzer = TextureAnalyzer(config.texture)
        extractor = FeatureExtractor(config, enhancer, analyzer)

        # Process frames
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Crop frame if needed
            if config.crop_start > 0:
                frame = frame[:, config.crop_start:]

            # Process frame
            extractor.process_frame(frame, frame_count)

            # Log progress
            frame_count += 1
            if frame_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

        logger.info(f"Processed {frame_count} frames")
        return True

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return False

    finally:
        if 'cap' in locals():
            cap.release()


def main() -> int:
    """Main function to run feature extraction.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Initialize configuration
        config = ProcessingConfig(
            crop_start=200,
            output_dir="07_03",
            save_format="png"
        )

        # Process video
        if len(sys.argv) != 2:
            logger.error("Usage: python 07_03_feature_extraction.py <video_path>")
            return 1

        video_path = sys.argv[1]
        if process_video(video_path, config):
            logger.info(f"Feature extraction completed. Results saved to {config.output_dir}")
            return 0
        else:
            logger.error("Feature extraction failed")
            return 1

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
