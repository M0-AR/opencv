"""Video Frame Comparison Module using SSIM.

This module compares consecutive video frames using Structural Similarity Index (SSIM)
to detect significant changes and save unique frames.

Algorithm Explanation for Beginners:
--------------------------------
We use a step-by-step process to find important changes in a video:

1. Frame Processing:
   - Read each frame from video
   - Convert to grayscale (black & white)
   - This makes comparison easier and faster

2. SSIM Comparison:
   - SSIM measures how similar two images are
   - Looks at patterns, not just pixel differences
   - Score ranges from 0 (different) to 1 (identical)
   - Like comparing two photos to spot changes

3. Change Detection:
   - If SSIM score is below threshold:
     * Frame has significant changes
     * Save it as a key frame
     * Find where changes occurred
   - If score is above threshold:
     * Frame is too similar
     * Skip it to save space

4. Visualization:
   - Show original frames
   - Highlight changed regions
   - Display difference maps
   - Create summary grid of key frames

Key Features:
- Automatic key frame extraction
- Interactive visualization
- Progress feedback
- Error handling
- Resource cleanup
- Performance optimization

Usage:
    python video_compare.py <video_path> [--threshold N] [--crop N]

Controls:
    'p' - Pause/Resume
    'n' - Next frame
    's' - Save current frame
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
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonParams:
    """Parameters for frame comparison."""
    ssim_threshold: float = 0.8
    crop_left: int = 200
    target_width: int = 640
    grid_cols: int = 3
    preview_size: Tuple[int, int] = (10, 5)
    save_comparison: bool = True


class VideoComparer:
    """A class to handle video frame comparison."""

    def __init__(self, output_dir: str = "compared_frames"):
        """Initialize the comparer.

        Args:
            output_dir: Base directory for output
        """
        self.output_dir = Path(output_dir)
        self.params = ComparisonParams()
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize state
        self.frame_count = 0
        self.saved_frames = []
        self.last_saved_frame = None
        self.paused = False

    def preprocess_frame(self, frame: NDArray) -> NDArray:
        """Preprocess frame for comparison.

        Args:
            frame: Input frame

        Returns:
            Preprocessed frame
        """
        try:
            # Crop frame
            if self.params.crop_left > 0:
                frame = frame[:, self.params.crop_left:]
                
            # Resize if needed
            if frame.shape[1] > self.params.target_width:
                height, width = frame.shape[:2]
                aspect_ratio = width / height
                target_height = int(self.params.target_width / aspect_ratio)
                frame = cv2.resize(
                    frame,
                    (self.params.target_width, target_height)
                )
                
            return frame
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {str(e)}")
            return frame

    def compare_frames(
        self,
        frame1: NDArray,
        frame2: NDArray
    ) -> Tuple[float, NDArray]:
        """Compare two frames using SSIM.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            Tuple of similarity score and difference map
        """
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Compare using SSIM
            score, diff = compare_ssim(gray1, gray2, full=True)
            
            # Convert difference to uint8
            diff = (diff * 255).astype(np.uint8)
            
            return score, diff
            
        except Exception as e:
            logger.error(f"Error comparing frames: {str(e)}")
            return 0.0, np.zeros_like(frame1[:, :, 0])

    def find_changes(self, diff: NDArray) -> List[Tuple[int, int, int, int]]:
        """Find regions of change in difference map.

        Args:
            diff: Difference map

        Returns:
            List of change regions (x, y, w, h)
        """
        try:
            # Threshold difference map
            thresh = cv2.threshold(
                diff,
                0,
                255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )[1]
            
            # Find contours
            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Get bounding rectangles
            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append((x, y, w, h))
                
            return regions
            
        except Exception as e:
            logger.error(f"Error finding changes: {str(e)}")
            return []

    def draw_changes(
        self,
        frame: NDArray,
        regions: List[Tuple[int, int, int, int]]
    ) -> NDArray:
        """Draw change regions on frame.

        Args:
            frame: Input frame
            regions: List of change regions

        Returns:
            Frame with changes highlighted
        """
        try:
            result = frame.copy()
            
            for x, y, w, h in regions:
                cv2.rectangle(
                    result,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )
                
            return result
            
        except Exception as e:
            logger.error(f"Error drawing changes: {str(e)}")
            return frame

    def save_frame(
        self,
        frame: NDArray,
        frame_number: int
    ) -> None:
        """Save frame to disk.

        Args:
            frame: Frame to save
            frame_number: Frame number
        """
        try:
            # Save frame
            output_path = self.output_dir / f"frame_{frame_number:04d}.jpg"
            cv2.imwrite(str(output_path), frame)
            
            # Update state
            self.saved_frames.append(frame)
            self.last_saved_frame = frame
            
            logger.info(f"Saved frame {frame_number}")
            
        except Exception as e:
            logger.error(f"Error saving frame: {str(e)}")

    def create_comparison_grid(
        self,
        frame1: NDArray,
        frame2: NDArray,
        diff: NDArray,
        regions: List[Tuple[int, int, int, int]]
    ) -> NDArray:
        """Create comparison grid visualization.

        Args:
            frame1: First frame
            frame2: Second frame
            diff: Difference map
            regions: Change regions

        Returns:
            Grid visualization
        """
        try:
            # Draw changes
            frame1_changes = self.draw_changes(frame1, regions)
            frame2_changes = self.draw_changes(frame2, regions)
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Create threshold image
            thresh = cv2.threshold(
                diff,
                0,
                255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )[1]
            
            # Stack images
            top_row = np.hstack((frame1_changes, frame2_changes))
            mid_row = np.hstack((
                cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
            ))
            bot_row = np.hstack((
                cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            ))
            
            return np.vstack((top_row, mid_row, bot_row))
            
        except Exception as e:
            logger.error(f"Error creating comparison grid: {str(e)}")
            return frame1

    def create_summary_grid(self) -> Optional[NDArray]:
        """Create summary grid of saved frames.

        Returns:
            Grid of saved frames
        """
        try:
            if not self.saved_frames:
                return None
                
            # Calculate grid size
            n_frames = len(self.saved_frames)
            n_cols = self.params.grid_cols
            n_rows = (n_frames - 1) // n_cols + 1
            
            # Get frame size
            height, width = self.saved_frames[0].shape[:2]
            
            # Create grid
            grid = np.zeros(
                (n_rows * height, n_cols * width, 3),
                dtype=np.uint8
            )
            
            # Fill grid
            for idx, frame in enumerate(self.saved_frames):
                row = idx // n_cols
                col = idx % n_cols
                y1 = row * height
                y2 = (row + 1) * height
                x1 = col * width
                x2 = (col + 1) * width
                grid[y1:y2, x1:x2] = frame
                
            return grid
            
        except Exception as e:
            logger.error(f"Error creating summary grid: {str(e)}")
            return None

    def process_frame(self, frame: NDArray) -> bool:
        """Process a single frame.

        Args:
            frame: Input frame

        Returns:
            True if frame was saved
        """
        try:
            # Preprocess frame
            frame = self.preprocess_frame(frame)
            
            # Initialize if first frame
            if self.last_saved_frame is None:
                self.save_frame(frame, self.frame_count)
                return True
            
            # Compare with last saved frame
            score, diff = self.compare_frames(
                self.last_saved_frame,
                frame
            )
            
            logger.info(
                f"Frame {self.frame_count}, "
                f"SSIM: {score:.3f}"
            )
            
            # Check if significant change
            if score < self.params.ssim_threshold:
                # Find changes
                regions = self.find_changes(diff)
                
                # Create comparison grid
                if self.params.save_comparison:
                    grid = self.create_comparison_grid(
                        self.last_saved_frame,
                        frame,
                        diff,
                        regions
                    )
                    
                    # Save comparison
                    output_path = self.output_dir / f"comparison_{self.frame_count:04d}.jpg"
                    cv2.imwrite(str(output_path), grid)
                
                # Save frame
                self.save_frame(frame, self.frame_count)
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return False

    def handle_key(self, key: int) -> bool:
        """Handle keyboard input.

        Args:
            key: Key code

        Returns:
            True if should continue, False if should quit
        """
        if key == ord('q'):
            return False
        elif key == ord('p'):
            self.paused = not self.paused
        elif key == ord('s'):
            # Force save current frame
            self.save_frame(
                self.last_saved_frame,
                self.frame_count
            )
        return True

    def process_video(self, video_path: str) -> bool:
        """Process video file.

        Args:
            video_path: Path to video file

        Returns:
            True if successful
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            
            logger.info(f"Processing video: {video_path}")
            
            while True:
                # Read frame
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Process frame
                    self.process_frame(frame)
                    self.frame_count += 1
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key(key):
                    break
            
            # Create summary grid
            logger.info("Creating summary grid...")
            grid = self.create_summary_grid()
            if grid is not None:
                cv2.imwrite(
                    str(self.output_dir / "summary.jpg"),
                    grid
                )
            
            logger.info(
                f"Processing complete. "
                f"Saved {len(self.saved_frames)} frames"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return False
            
        finally:
            cap.release()
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
            '--threshold',
            type=float,
            default=0.8,
            help='SSIM threshold'
        )
        parser.add_argument(
            '--crop',
            type=int,
            default=200,
            help='Pixels to crop from left'
        )
        
        args = parser.parse_args()
        
        # Create comparer
        comparer = VideoComparer()
        comparer.params.ssim_threshold = args.threshold
        comparer.params.crop_left = args.crop
        
        # Process video
        if comparer.process_video(args.video_path):
            return 0
        else:
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    plt.switch_backend('TkAgg')
    sys.exit(main())
