"""Video Frame Comparison Module using SSIM.

This module compares multiple frames from a video using the Structural
Similarity Index (SSIM) to detect and visualize changes over time.

Algorithm Explanation for Beginners:
--------------------------------
We analyze video changes through these steps:

1. Frame Extraction:
   - Read frames from video file
   - Like taking snapshots from a movie
   - Convert to grayscale:
     * Simplifies comparison
     * Focuses on structure not color

2. SSIM Comparison:
   - Compare frames with a reference
   - Like spot-the-difference game
   - SSIM measures:
     * Brightness similarity
     * Contrast similarity
     * Structure similarity
   - Score ranges 0 (different) to 1 (identical)

3. Difference Detection:
   - Create difference map
   - Like subtracting two images
   - White shows changes
   - Black shows unchanged areas

4. Change Highlighting:
   - Find regions of change
   - Draw rectangles around changes
   - Like circling differences
   - Makes changes easy to spot

5. Result Visualization:
   - Show multiple views:
     * Original frames
     * Grayscale versions
     * Difference maps
     * Highlighted changes
   - Helps understand the changes

Key Features:
- Interactive frame selection
- Real-time comparison
- Visual change detection
- Progress feedback
- Error handling
- Resource cleanup

Usage:
    python video_compare.py <video_path> [--reference N] [--frames N N N]

Controls:
    'n' - Next frame set
    'p' - Previous frame set
    's' - Save comparison
    'q' - Quit
"""

from dataclasses import dataclass
import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity as compare_ssim


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonParams:
    """Parameters for video frame comparison."""
    reference_frame: int = 0
    crop_start: int = 200
    min_contour_area: int = 100
    threshold_min: int = 30
    contour_color: Tuple[int, int, int] = (0, 255, 0)
    contour_thickness: int = 2
    display_size: Tuple[int, int] = (1280, 720)


class VideoComparer:
    """A class to handle video frame comparison."""

    def __init__(self, output_dir: str = "comparison_output"):
        """Initialize the comparer.

        Args:
            output_dir: Base directory for output
        """
        self.output_dir = Path(output_dir)
        self.params = ComparisonParams()
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize state
        self.video = None
        self.frames = []
        self.gray_frames = []
        self.differences = []
        self.highlighted = []
        
        # Set up display
        matplotlib.use('TkAgg')

    def open_video(self, path: str) -> bool:
        """Open video file.

        Args:
            path: Path to video file

        Returns:
            True if successful
        """
        try:
            self.video = cv2.VideoCapture(path)
            if not self.video.isOpened():
                raise RuntimeError(f"Could not open video: {path}")
            
            logger.info("Video opened successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error opening video: {str(e)}")
            return False

    def read_frame(self, index: int) -> Optional[NDArray]:
        """Read specific frame from video.

        Args:
            index: Frame index

        Returns:
            Frame if successful
        """
        try:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = self.video.read()
            
            if not ret:
                raise RuntimeError(f"Could not read frame {index}")
            
            # Crop frame
            if self.params.crop_start > 0:
                frame = frame[:, self.params.crop_start:]
            
            return frame
            
        except Exception as e:
            logger.error(f"Error reading frame {index}: {str(e)}")
            return None

    def load_frames(self, indices: List[int]) -> bool:
        """Load frames from video.

        Args:
            indices: List of frame indices

        Returns:
            True if successful
        """
        try:
            # Reset state
            self.frames = []
            self.gray_frames = []
            
            # Load frames
            for idx in indices:
                frame = self.read_frame(idx)
                if frame is None:
                    return False
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                self.frames.append(frame)
                self.gray_frames.append(gray)
            
            logger.info(f"Loaded {len(self.frames)} frames")
            return True
            
        except Exception as e:
            logger.error(f"Error loading frames: {str(e)}")
            return False

    def compare_frames(self) -> bool:
        """Compare frames with reference frame.

        Returns:
            True if successful
        """
        try:
            if not self.frames:
                raise RuntimeError("No frames loaded")
            
            # Reset state
            self.differences = []
            self.highlighted = []
            
            # Get reference frame
            ref_gray = self.gray_frames[self.params.reference_frame]
            ref_frame = self.frames[self.params.reference_frame]
            
            # Compare each frame
            for i, (frame, gray) in enumerate(zip(
                self.frames,
                self.gray_frames
            )):
                if i == self.params.reference_frame:
                    continue
                
                # Compare with SSIM
                score, diff = compare_ssim(ref_gray, gray, full=True)
                diff = (diff * 255).astype(np.uint8)
                
                # Threshold difference
                thresh = cv2.threshold(
                    diff,
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
                
                # Draw contours
                ref_highlighted = ref_frame.copy()
                frame_highlighted = frame.copy()
                
                for contour in contours:
                    if cv2.contourArea(contour) > self.params.min_contour_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(
                            ref_highlighted,
                            (x, y),
                            (x + w, y + h),
                            self.params.contour_color,
                            self.params.contour_thickness
                        )
                        cv2.rectangle(
                            frame_highlighted,
                            (x, y),
                            (x + w, y + h),
                            self.params.contour_color,
                            self.params.contour_thickness
                        )
                
                self.differences.append((score, diff, thresh))
                self.highlighted.append((ref_highlighted, frame_highlighted))
                
                logger.info(f"Frame {i} SSIM: {score:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error comparing frames: {str(e)}")
            return False

    def display_comparison(self, index: int) -> None:
        """Display comparison results.

        Args:
            index: Index of comparison to display
        """
        try:
            if not self.differences or not self.highlighted:
                raise RuntimeError("No comparison results")
            
            if index >= len(self.differences):
                raise RuntimeError("Invalid comparison index")
            
            # Get comparison data
            score, diff, thresh = self.differences[index]
            ref_high, frame_high = self.highlighted[index]
            
            # Create figure
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            
            # Show images
            axs[0, 0].imshow(cv2.cvtColor(ref_high, cv2.COLOR_BGR2RGB))
            axs[0, 0].set_title("Reference Frame")
            axs[0, 0].axis('off')
            
            axs[0, 1].imshow(cv2.cvtColor(frame_high, cv2.COLOR_BGR2RGB))
            axs[0, 1].set_title(f"Compared Frame (SSIM: {score:.3f})")
            axs[0, 1].axis('off')
            
            axs[0, 2].imshow(self.gray_frames[self.params.reference_frame], cmap='gray')
            axs[0, 2].set_title("Reference Gray")
            axs[0, 2].axis('off')
            
            axs[1, 0].imshow(self.gray_frames[index + 1], cmap='gray')
            axs[1, 0].set_title("Compared Gray")
            axs[1, 0].axis('off')
            
            axs[1, 1].imshow(diff, cmap='gray')
            axs[1, 1].set_title("Difference Map")
            axs[1, 1].axis('off')
            
            axs[1, 2].imshow(thresh, cmap='gray')
            axs[1, 2].set_title("Threshold Map")
            axs[1, 2].axis('off')
            
            # Show plot
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error displaying comparison: {str(e)}")

    def save_comparison(
        self,
        index: int,
        prefix: str = "comparison"
    ) -> bool:
        """Save comparison results.

        Args:
            index: Index of comparison to save
            prefix: Prefix for output files

        Returns:
            True if successful
        """
        try:
            if not self.differences or not self.highlighted:
                raise RuntimeError("No comparison results")
            
            if index >= len(self.differences):
                raise RuntimeError("Invalid comparison index")
            
            # Get comparison data
            score, diff, thresh = self.differences[index]
            ref_high, frame_high = self.highlighted[index]
            
            # Create output paths
            paths = {
                'reference': self.output_dir / f"{prefix}_reference.jpg",
                'compared': self.output_dir / f"{prefix}_compared.jpg",
                'diff': self.output_dir / f"{prefix}_diff.jpg",
                'thresh': self.output_dir / f"{prefix}_thresh.jpg"
            }
            
            # Save images
            cv2.imwrite(str(paths['reference']), ref_high)
            cv2.imwrite(str(paths['compared']), frame_high)
            cv2.imwrite(str(paths['diff']), diff)
            cv2.imwrite(str(paths['thresh']), thresh)
            
            logger.info(f"Saved comparison to {self.output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving comparison: {str(e)}")
            return False

    def run(
        self,
        video_path: str,
        frame_indices: List[int]
    ) -> bool:
        """Run comparison workflow.

        Args:
            video_path: Path to video file
            frame_indices: List of frame indices

        Returns:
            True if successful
        """
        try:
            # Open video
            if not self.open_video(video_path):
                return False
            
            # Load frames
            if not self.load_frames(frame_indices):
                return False
            
            # Compare frames
            if not self.compare_frames():
                return False
            
            # Display comparisons
            for i in range(len(self.differences)):
                self.display_comparison(i)
                
                # Handle keyboard input
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_comparison(i)
            
            return True
            
        except Exception as e:
            logger.error(f"Error running comparer: {str(e)}")
            return False
            
        finally:
            if self.video is not None:
                self.video.release()
            cv2.destroyAllWindows()


def main() -> int:
    """Main function.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(
            description='Compare video frames using SSIM'
        )
        parser.add_argument(
            'video_path',
            help='Path to video file'
        )
        parser.add_argument(
            '--reference',
            type=int,
            default=0,
            help='Reference frame index'
        )
        parser.add_argument(
            '--frames',
            type=int,
            nargs='+',
            default=[0, 100, 1000],
            help='Frame indices to compare'
        )
        
        args = parser.parse_args()
        
        # Create comparer
        comparer = VideoComparer()
        comparer.params.reference_frame = args.reference
        
        # Run comparison
        if not comparer.run(args.video_path, args.frames):
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())