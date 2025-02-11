"""Template Matching Module.

This module demonstrates real-time template matching in video streams, detecting and
highlighting regions in frames that match a specified pattern or template.

Algorithm Explanation for Beginners:
--------------------------------
Template Matching is like finding a small picture within a bigger picture. Here's how it works:

1. Template Creation:
   - We take the first frame of the video
   - Convert it to grayscale (black and white)
   - This becomes our "template" - the pattern we want to find
   
2. Frame Processing:
   - For each new frame in the video:
     * Convert it to grayscale
     * Slide the template over every possible position
     * Calculate how well the template matches at each position
   
3. Match Detection:
   - For each position, we get a similarity score (0 to 1)
   - 1 means perfect match, 0 means no match
   - If the score is above our threshold, we found a match!
   - We draw a green rectangle around matched regions

The result shows:
- Left: Original frame
- Right: Frame with detected matches highlighted
- Green rectangles show where matches were found

This helps us track specific patterns or objects throughout the video.

Key Features:
- Real-time template matching
- Adjustable matching sensitivity
- Side-by-side comparison
- Frame saving capability
- Progress tracking

Usage:
    python 08_Template_Matching.py <video_path>

Controls:
    'q' - quit
    's' - save current frame
    '+' - increase matching sensitivity
    '-' - decrease matching sensitivity
    'r' - reset parameters to default
    'h' - toggle histogram equalization
"""

import logging
import os
from pathlib import Path
import sys
from typing import Tuple, Optional
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
class MatchingParams:
    """Parameters for template matching."""
    threshold: float = 0.8  # Matching threshold (0-1)
    min_threshold: float = 0.5
    max_threshold: float = 0.95
    use_histogram: bool = False
    crop_start: int = 200  # Starting point for cropping
    
    def adjust_sensitivity(self, increase: bool = True) -> None:
        """Adjust matching sensitivity.
        
        Args:
            increase: If True, increase sensitivity; if False, decrease
        """
        step = 0.05
        if increase and self.threshold > self.min_threshold:
            self.threshold = max(self.threshold - step, self.min_threshold)
        elif not increase and self.threshold < self.max_threshold:
            self.threshold = min(self.threshold + step, self.max_threshold)
            
    def reset(self) -> None:
        """Reset parameters to default values."""
        self.__init__()


class TemplateMatcher:
    """A class to handle template matching and visualization."""

    def __init__(self, output_dir: str = "template_matches"):
        """Initialize the matcher.

        Args:
            output_dir: Directory to save output images
        """
        self.output_dir = Path(output_dir)
        self.frame_count = 0
        self.match_count = 0
        self.window_name = 'Original and Template Matching Results'
        self.params = MatchingParams()
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

    def load_video(self, video_path: str) -> Tuple[Optional[cv2.VideoCapture], bool]:
        """Load video file.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (VideoCapture object, success flag)
        """
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("Could not open video file")
                
            return cap, True
            
        except Exception as e:
            logger.error(f"Error loading video: {str(e)}")
            return None, False

    def get_template(self, cap: cv2.VideoCapture) -> Tuple[Optional[NDArray], Optional[Tuple[int, int]]]:
        """Extract template from first frame.

        Args:
            cap: VideoCapture object

        Returns:
            Tuple of (template image, template dimensions)
        """
        try:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Could not read first frame")
                
            # Process frame
            frame = frame[:, self.params.crop_start:]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.params.use_histogram:
                gray = cv2.equalizeHist(gray)
                
            return gray, gray.shape[::-1]  # width, height
            
        except Exception as e:
            logger.error(f"Error extracting template: {str(e)}")
            return None, None

    def find_matches(
        self,
        frame: NDArray,
        template: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """Find template matches in frame.

        Args:
            frame: Input frame
            template: Template to match

        Returns:
            Tuple of (matched locations, match scores)
        """
        # Template matching
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= self.params.threshold)
        
        return locations, result

    def draw_matches(
        self,
        frame: NDArray,
        locations: NDArray,
        template_size: Tuple[int, int]
    ) -> NDArray:
        """Draw rectangles around matches.

        Args:
            frame: Input frame
            locations: Match locations
            template_size: Size of template

        Returns:
            Frame with matches drawn
        """
        result = frame.copy()
        w, h = template_size
        
        for pt in zip(*locations[::-1]):
            cv2.rectangle(
                result,
                pt,
                (pt[0] + w, pt[1] + h),
                (0, 255, 0),
                2
            )
            
        return result

    def create_display_frame(
        self,
        original: NDArray,
        matched: NDArray,
        target_width: int = 1400,
        target_height: int = 700
    ) -> NDArray:
        """Create display frame with both views.

        Args:
            original: Original frame
            matched: Frame with matches
            target_width: Desired width
            target_height: Desired height

        Returns:
            Combined display frame
        """
        # Combine side by side
        combined = cv2.hconcat([original, matched])
        
        # Resize to target dimensions
        return cv2.resize(combined, (target_width, target_height))

    def save_frame(self, frame: NDArray) -> None:
        """Save current frame.

        Args:
            frame: Frame to save
        """
        filename = self.output_dir / f"frame_{self.frame_count:04d}.jpg"
        cv2.imwrite(str(filename), frame)
        logger.info(f"Saved frame to: {filename}")

    def process_video(self, video_path: str) -> bool:
        """Process video with template matching.

        Args:
            video_path: Path to video file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load video
            cap, success = self.load_video(video_path)
            if not success:
                return False

            # Get template
            template, dimensions = self.get_template(cap)
            if template is None or dimensions is None:
                return False

            logger.info("Starting video processing...")
            logger.info("Controls:")
            logger.info("  'q' - quit")
            logger.info("  's' - save current frame")
            logger.info("  '+' - increase matching sensitivity")
            logger.info("  '-' - decrease matching sensitivity")
            logger.info("  'r' - reset parameters")
            logger.info("  'h' - toggle histogram equalization")

            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                frame = frame[:, self.params.crop_start:]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if self.params.use_histogram:
                    gray = cv2.equalizeHist(gray)

                # Find and draw matches
                locations, scores = self.find_matches(gray, template)
                matched_frame = self.draw_matches(frame, locations, dimensions)
                
                # Update match count
                self.match_count = len(locations[0])
                
                # Create display frame
                display_frame = self.create_display_frame(frame, matched_frame)

                # Add parameter info
                hist = "On" if self.params.use_histogram else "Off"
                cv2.putText(
                    display_frame,
                    f"Thresh: {self.params.threshold:.2f} | "
                    f"Matches: {self.match_count} | "
                    f"Hist: {hist}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                # Show result
                cv2.imshow(self.window_name, display_frame)

                # Handle keyboard input
                key = cv2.waitKey(25) & 0xFF
                if key == ord('q'):
                    logger.info("Quitting...")
                    break
                elif key == ord('s'):
                    self.save_frame(display_frame)
                elif key == ord('+'):
                    self.params.adjust_sensitivity(increase=True)
                    logger.info(f"Increased sensitivity: {self.params.threshold:.2f}")
                elif key == ord('-'):
                    self.params.adjust_sensitivity(increase=False)
                    logger.info(f"Decreased sensitivity: {self.params.threshold:.2f}")
                elif key == ord('r'):
                    self.params.reset()
                    logger.info("Reset parameters to default")
                elif key == ord('h'):
                    self.params.use_histogram = not self.params.use_histogram
                    hist = "enabled" if self.params.use_histogram else "disabled"
                    logger.info(f"Histogram equalization {hist}")

                self.frame_count += 1

            logger.info(f"Total frames processed: {self.frame_count}")
            return True

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return False

        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()


def main() -> int:
    """Main function.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Check command line arguments
        if len(sys.argv) != 2:
            logger.error("Usage: python 08_Template_Matching.py <video_path>")
            return 1

        video_path = sys.argv[1]
        
        # Create matcher and process video
        matcher = TemplateMatcher()
        if matcher.process_video(video_path):
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
