"""Image Similarity Analysis Module using SSIM.

This module implements Structural Similarity Index (SSIM) analysis between
two images to detect and visualize differences. It includes functionality for:
- Loading and preprocessing images
- Computing SSIM scores
- Detecting and highlighting differences
- Visualizing results

Algorithm Explanation for Beginners:
--------------------------------
SSIM (Structural Similarity Index) works by comparing images in 3 ways:

1. Luminance (Brightness):
   - Like comparing how bright or dark two photos are
   - Helps spot lighting changes
   - Not affected by small pixel differences

2. Contrast:
   - Like comparing the range of light to dark
   - Helps find areas that got blurrier or sharper
   - Looks at patterns, not exact pixels

3. Structure:
   - Like comparing the shapes and edges
   - Helps find moved or changed objects
   - More like how human eyes work

The Process:
1. Image Loading:
   - Read both images
   - Convert to grayscale (black & white)
   - Make sure they're the same size

2. SSIM Calculation:
   - Look at small windows (patches) of both images
   - Compare brightness, contrast, structure
   - Combine into one score (1.0 = identical)

3. Difference Detection:
   - Create a difference map
   - White = big differences
   - Black = small differences

4. Visualization:
   - Draw boxes around changed areas
   - Show original, modified, and difference
   - Display SSIM score

Interactive Controls:
- 'q' - Quit
- 's' - Save current comparison
- '+' - Increase sensitivity
- '-' - Decrease sensitivity
- 'r' - Reset parameters
- 'h' - Show/hide help

Usage:
    python ssim_analyzer.py <image1_path> <image2_path> [--target-size width height]
"""

from dataclasses import dataclass
import argparse
import logging
from pathlib import Path
import sys
from typing import List, Tuple, Optional, Union, Dict

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure matplotlib backend
try:
    matplotlib.use('TkAgg')
except Exception as e:
    logger.warning(f"Failed to set matplotlib backend: {str(e)}")


@dataclass
class AnalysisParams:
    """Parameters for image analysis."""
    win_size: int = 7
    threshold_min: int = 30
    min_contour_area: int = 100
    box_color: Tuple[int, int, int] = (0, 255, 0)
    box_thickness: int = 2
    display_size: Tuple[int, int] = (1280, 720)
    show_help: bool = True


@dataclass
class ImageComparisonResult:
    """Data class to store image comparison results."""
    ssim_score: float
    difference_map: NDArray
    thresholded_image: NDArray
    contours: List[np.ndarray]
    annotated_image: NDArray


class ImageAnalyzer:
    """Class to handle image similarity analysis."""

    def __init__(self, output_dir: str = "comparison_output"):
        """Initialize analyzer.

        Args:
            output_dir: Directory for saving results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.params = AnalysisParams()
        self._setup_display()

    def _setup_display(self) -> None:
        """Set up display parameters."""
        cv2.namedWindow('Analysis', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Analysis', *self.params.display_size)

    def _draw_help(self, image: NDArray) -> NDArray:
        """Draw help text on image.

        Args:
            image: Input image

        Returns:
            Image with help text
        """
        if not self.params.show_help:
            return image

        help_text = [
            "Controls:",
            "q - Quit",
            "s - Save comparison",
            "+ - Increase sensitivity",
            "- - Decrease sensitivity",
            "r - Reset parameters",
            "h - Show/hide help"
        ]

        result = image.copy()
        y = 30
        for text in help_text:
            cv2.putText(
                result,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            y += 25

        return result

    def _draw_parameters(self, image: NDArray) -> NDArray:
        """Draw current parameters on image.

        Args:
            image: Input image

        Returns:
            Image with parameter text
        """
        param_text = [
            f"Window Size: {self.params.win_size}",
            f"Threshold: {self.params.threshold_min}",
            f"Min Area: {self.params.min_contour_area}"
        ]

        result = image.copy()
        y = 30
        for text in param_text:
            cv2.putText(
                result,
                text,
                (result.shape[1] - 200, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            y += 25

        return result

    def load_image(
        self,
        image_path: str,
        color_mode: int = cv2.IMREAD_COLOR
    ) -> Optional[NDArray]:
        """Load an image from file.

        Args:
            image_path: Path to the image file
            color_mode: OpenCV color mode flag

        Returns:
            Loaded image or None if failed
        """
        try:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = cv2.imread(image_path, color_mode)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            return image

        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None

    def preprocess_images(
        self,
        img1: NDArray,
        img2: NDArray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[Optional[NDArray], Optional[NDArray]]:
        """Preprocess images for comparison.

        Args:
            img1: First input image
            img2: Second input image
            target_size: Optional target size

        Returns:
            Tuple of preprocessed images
        """
        try:
            if img1 is None or img2 is None:
                raise ValueError("Input images cannot be None")

            # Resize if needed
            if target_size:
                img1 = cv2.resize(img1, target_size)
                img2 = cv2.resize(img2, target_size)
            elif img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            return gray1, gray2

        except Exception as e:
            logger.error(f"Error preprocessing images: {str(e)}")
            return None, None

    def compute_ssim(
        self,
        img1: NDArray,
        img2: NDArray
    ) -> Tuple[Optional[float], Optional[NDArray]]:
        """Compute SSIM between images.

        Args:
            img1: First grayscale image
            img2: Second grayscale image

        Returns:
            Tuple of (SSIM score, difference map)
        """
        try:
            if img1.shape != img2.shape:
                raise ValueError("Images must have same dimensions")

            score, diff = compare_ssim(
                img1,
                img2,
                full=True,
                win_size=self.params.win_size
            )
            diff = (diff * 255).astype("uint8")

            return score, diff

        except Exception as e:
            logger.error(f"Error computing SSIM: {str(e)}")
            return None, None

    def detect_differences(
        self,
        diff_image: NDArray
    ) -> Tuple[Optional[NDArray], Optional[List[np.ndarray]]]:
        """Detect differences in image.

        Args:
            diff_image: Difference image

        Returns:
            Tuple of (threshold image, contours)
        """
        try:
            if diff_image is None:
                raise ValueError("Invalid difference image")

            # Threshold
            thresh = cv2.threshold(
                diff_image,
                self.params.threshold_min,
                255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )[1]

            # Find contours
            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Filter small contours
            contours = [
                c for c in contours
                if cv2.contourArea(c) > self.params.min_contour_area
            ]

            return thresh, contours

        except Exception as e:
            logger.error(f"Error detecting differences: {str(e)}")
            return None, None

    def draw_differences(
        self,
        image: NDArray,
        contours: List[np.ndarray]
    ) -> Optional[NDArray]:
        """Draw difference boxes on image.

        Args:
            image: Input image
            contours: List of contours

        Returns:
            Annotated image
        """
        try:
            result = image.copy()

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(
                    result,
                    (x, y),
                    (x + w, y + h),
                    self.params.box_color,
                    self.params.box_thickness
                )

            return result

        except Exception as e:
            logger.error(f"Error drawing differences: {str(e)}")
            return None

    def save_results(
        self,
        result: ImageComparisonResult,
        prefix: str = "comparison"
    ) -> bool:
        """Save analysis results.

        Args:
            result: Comparison results
            prefix: Output filename prefix

        Returns:
            True if successful
        """
        try:
            paths = {
                'annotated': self.output_dir / f"{prefix}_annotated.jpg",
                'difference': self.output_dir / f"{prefix}_difference.jpg",
                'threshold': self.output_dir / f"{prefix}_threshold.jpg"
            }

            cv2.imwrite(str(paths['annotated']), result.annotated_image)
            cv2.imwrite(str(paths['difference']), result.difference_map)
            cv2.imwrite(str(paths['threshold']), result.thresholded_image)

            logger.info(f"Saved results to {self.output_dir}")
            return True

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False

    def analyze_images(
        self,
        image1_path: str,
        image2_path: str,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Optional[ImageComparisonResult]:
        """Analyze image similarity.

        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            target_size: Optional target size

        Returns:
            Analysis results
        """
        try:
            # Load images
            img1 = self.load_image(image1_path)
            img2 = self.load_image(image2_path)
            if img1 is None or img2 is None:
                return None

            # Preprocess
            gray1, gray2 = self.preprocess_images(img1, img2, target_size)
            if gray1 is None or gray2 is None:
                return None

            # Compute SSIM
            score, diff = self.compute_ssim(gray1, gray2)
            if score is None or diff is None:
                return None

            # Detect differences
            thresh, contours = self.detect_differences(diff)
            if thresh is None or contours is None:
                return None

            # Draw differences
            result = self.draw_differences(img1, contours)
            if result is None:
                return None

            # Create result object
            comparison = ImageComparisonResult(
                ssim_score=score,
                difference_map=diff,
                thresholded_image=thresh,
                contours=contours,
                annotated_image=result
            )

            return comparison

        except Exception as e:
            logger.error(f"Error analyzing images: {str(e)}")
            return None

    def display_analysis(
        self,
        img1: NDArray,
        img2: NDArray,
        result: ImageComparisonResult
    ) -> None:
        """Display analysis results.

        Args:
            img1: First image
            img2: Second image
            result: Analysis results
        """
        try:
            while True:
                # Create display
                display = np.hstack([
                    cv2.resize(img1, (640, 480)),
                    cv2.resize(img2, (640, 480))
                ])
                display = np.vstack([
                    display,
                    np.hstack([
                        cv2.resize(result.difference_map, (640, 480)),
                        cv2.resize(result.annotated_image, (640, 480))
                    ])
                ])

                # Add text
                cv2.putText(
                    display,
                    f"SSIM Score: {result.ssim_score:.3f}",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2
                )

                # Add help and parameters
                display = self._draw_help(display)
                display = self._draw_parameters(display)

                # Show display
                cv2.imshow('Analysis', display)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_results(result)
                elif key == ord('+'):
                    self.params.threshold_min = min(255, self.params.threshold_min + 5)
                elif key == ord('-'):
                    self.params.threshold_min = max(0, self.params.threshold_min - 5)
                elif key == ord('r'):
                    self.params = AnalysisParams()
                elif key == ord('h'):
                    self.params.show_help = not self.params.show_help

        except Exception as e:
            logger.error(f"Error displaying analysis: {str(e)}")

        finally:
            cv2.destroyAllWindows()


def main() -> int:
    """Main function.

    Returns:
        Exit code
    """
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(
            description='Analyze image similarity using SSIM'
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
            '--target-size',
            nargs=2,
            type=int,
            metavar=('WIDTH', 'HEIGHT'),
            help='Target size for image resizing'
        )

        args = parser.parse_args()

        # Create analyzer
        analyzer = ImageAnalyzer()

        # Load and analyze images
        target_size = tuple(args.target_size) if args.target_size else None
        result = analyzer.analyze_images(
            args.image1,
            args.image2,
            target_size
        )

        if result is None:
            logger.error("Analysis failed")
            return 1

        # Display results
        img1 = analyzer.load_image(args.image1)
        img2 = analyzer.load_image(args.image2)
        if img1 is None or img2 is None:
            return 1

        analyzer.display_analysis(img1, img2, result)
        return 0

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())