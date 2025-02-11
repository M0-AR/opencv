"""Template Matching with Edge Detection Module.

This module demonstrates advanced frame analysis using edge detection and structural
similarity (SSIM) to identify unique frames in a video stream. It combines edge
detection, contour analysis, and Hu Moments for comprehensive shape analysis.

Algorithm Explanation for Beginners:
--------------------------------
This program uses multiple techniques to find unique and interesting frames:

1. Edge Detection:
   - We convert each frame to grayscale (black and white)
   - We use the Canny algorithm to find edges (sharp changes in brightness)
   - Edges help us identify the shapes and objects in the frame
   
2. Structural Similarity (SSIM):
   - We compare each new frame's edges with previously saved ones
   - SSIM measures how similar two images are (0 = different, 1 = identical)
   - If a frame is different enough (SSIM < threshold), we consider it unique
   
3. Contour Analysis:
   - We find contours (outlines) in the edge image
   - For each contour, we calculate Hu Moments
   - Hu Moments are special numbers that describe shape properties
   - They help us understand the shapes we've detected

The result shows:
- Left: Original grayscale frame
- Right: Detected contours
- Saved frames are stored in the 'unique_frames' directory

This helps us identify and save frames that show different or interesting content.

Key Features:
- Real-time edge detection
- Structural similarity comparison
- Contour detection and analysis
- Hu Moments calculation
- Frame saving capability
- Progress tracking

Usage:
    python 08_Template_Matching_uniqe_edge.py <video_path>

Controls:
    'q' - quit
    's' - save current frame
    '+' - increase detection sensitivity
    '-' - decrease detection sensitivity
    'r' - reset parameters to default
    'c' - toggle contour display
    'h' - toggle histogram equalization
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
from skimage.metrics import structural_similarity as ssim


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DetectionParams:
    """Parameters for frame detection and analysis."""
    similarity_threshold: float = 0.8  # SSIM threshold (0-1)
    canny_low: int = 50  # Lower threshold for Canny
    canny_high: int = 150  # Upper threshold for Canny
    min_contour_area: float = 100.0  # Minimum contour area
    max_contour_area: float = 10000.0  # Maximum contour area
    use_histogram: bool = True
    show_contours: bool = True
    
    def adjust_sensitivity(self, increase: bool = True) -> None:
        """Adjust detection sensitivity.
        
        Args:
            increase: If True, increase sensitivity; if False, decrease
        """
        factor = 0.9 if increase else 1.1  # Inverse for thresholds
        self.similarity_threshold *= factor
        self.canny_low = int(self.canny_low * factor)
        self.canny_high = int(self.canny_high * factor)
        
    def reset(self) -> None:
        """Reset parameters to default values."""
        self.__init__()


class UniqueFrameDetector:
    """A class to handle unique frame detection and visualization."""

    def __init__(self, output_dir: str = "unique_frames"):
        """Initialize the detector.

        Args:
            output_dir: Directory to save unique frames
        """
        self.output_dir = Path(output_dir)
        self.frame_count = 0
        self.unique_count = 0
        self.window_name = 'Original and Contours'
        self.params = DetectionParams()
        self.unique_edges: List[NDArray] = []
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

    def preprocess_frame(self, frame: NDArray) -> NDArray:
        """Preprocess frame before analysis.

        Args:
            frame: Input frame

        Returns:
            Preprocessed grayscale frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Remove black area if present
        gray = gray[:, 200:]
        
        # Apply histogram equalization if enabled
        if self.params.use_histogram:
            gray = cv2.equalizeHist(gray)
            
        return gray

    def detect_edges(self, frame: NDArray) -> NDArray:
        """Detect edges in the frame.

        Args:
            frame: Input grayscale frame

        Returns:
            Edge detection result
        """
        return cv2.Canny(
            frame,
            self.params.canny_low,
            self.params.canny_high
        )

    def is_frame_unique(self, edges: NDArray) -> bool:
        """Check if frame is unique using SSIM.

        Args:
            edges: Edge detection result

        Returns:
            True if frame is unique, False otherwise
        """
        for unique_edge in self.unique_edges:
            # Skip if shapes don't match
            if edges.shape != unique_edge.shape:
                continue
                
            similarity_index, _ = ssim(edges, unique_edge, full=True)
            if similarity_index > self.params.similarity_threshold:
                return False
                
        return True

    def find_contours(self, edges: NDArray) -> Tuple[List[NDArray], NDArray]:
        """Find contours in edge image.

        Args:
            edges: Edge detection result

        Returns:
            Tuple of (filtered contours, contour image)
        """
        # Find all contours
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area
        filtered_contours = []
        contour_img = np.zeros_like(edges)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.params.min_contour_area <= area <= self.params.max_contour_area:
                filtered_contours.append(contour)
                
        return filtered_contours, contour_img

    def calculate_hu_moments(self, contour: NDArray) -> NDArray:
        """Calculate Hu Moments for a contour.

        Args:
            contour: Input contour

        Returns:
            Array of Hu Moments
        """
        moments = cv2.moments(contour)
        return cv2.HuMoments(moments)

    def draw_contours(
        self,
        image: NDArray,
        contours: List[NDArray]
    ) -> NDArray:
        """Draw contours and Hu Moments.

        Args:
            image: Input image
            contours: List of contours

        Returns:
            Image with contours drawn
        """
        result = image.copy()
        
        if self.params.show_contours:
            for contour in contours:
                # Draw contour
                cv2.drawContours(result, [contour], -1, 255, 2)
                
                # Calculate and draw Hu Moments
                hu_moments = self.calculate_hu_moments(contour)
                
                # Get contour center for text placement
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Format first Hu Moment
                    text = f"Hu: {hu_moments[0][0]:.2e}"
                    cv2.putText(
                        result,
                        text,
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        255,
                        1
                    )
        
        return result

    def create_display_frame(
        self,
        original: NDArray,
        contours: List[NDArray],
        target_width: int = 1400,
        target_height: int = 700
    ) -> NDArray:
        """Create display frame with both views.

        Args:
            original: Original frame
            contours: List of contours
            target_width: Desired width
            target_height: Desired height

        Returns:
            Combined display frame
        """
        # Create contour visualization
        contour_img = np.zeros_like(original)
        contour_display = self.draw_contours(contour_img, contours)
        
        # Convert to BGR for display
        original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        contour_bgr = cv2.cvtColor(contour_display, cv2.COLOR_GRAY2BGR)
        
        # Combine side by side
        combined = cv2.hconcat([original_bgr, contour_bgr])
        
        # Resize to target dimensions
        return cv2.resize(combined, (target_width, target_height))

    def save_frame(self, frame: NDArray) -> None:
        """Save unique frame.

        Args:
            frame: Frame to save
        """
        filename = self.output_dir / f"unique_frame_{self.unique_count:04d}.jpg"
        cv2.imwrite(str(filename), frame)
        logger.info(f"Saved unique frame to: {filename}")
        self.unique_count += 1

    def process_video(self, video_path: str) -> bool:
        """Process video and detect unique frames.

        Args:
            video_path: Path to input video

        Returns:
            True if successful, False otherwise
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error("Could not open video file")
                return False

            logger.info("Starting video processing...")
            logger.info("Controls:")
            logger.info("  'q' - quit")
            logger.info("  's' - save current frame")
            logger.info("  '+' - increase detection sensitivity")
            logger.info("  '-' - decrease detection sensitivity")
            logger.info("  'r' - reset parameters")
            logger.info("  'c' - toggle contour display")
            logger.info("  'h' - toggle histogram equalization")

            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                gray = self.preprocess_frame(frame)
                edges = self.detect_edges(gray)
                
                # Check if frame is unique
                if self.is_frame_unique(edges):
                    self.unique_edges.append(edges)
                    self.save_frame(frame)

                # Find and analyze contours
                contours, _ = self.find_contours(edges)
                
                # Create display frame
                display_frame = self.create_display_frame(gray, contours)

                # Add parameter info
                hist = "On" if self.params.use_histogram else "Off"
                cont = "On" if self.params.show_contours else "Off"
                cv2.putText(
                    display_frame,
                    f"Thresh: {self.params.similarity_threshold:.2f} | "
                    f"Unique: {self.unique_count} | "
                    f"Hist: {hist} | Cont: {cont}",
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
                    self.save_frame(frame)
                elif key == ord('+'):
                    self.params.adjust_sensitivity(increase=True)
                    logger.info(f"Increased sensitivity: {self.params.similarity_threshold:.2f}")
                elif key == ord('-'):
                    self.params.adjust_sensitivity(increase=False)
                    logger.info(f"Decreased sensitivity: {self.params.similarity_threshold:.2f}")
                elif key == ord('r'):
                    self.params.reset()
                    logger.info("Reset parameters to default")
                elif key == ord('c'):
                    self.params.show_contours = not self.params.show_contours
                    cont = "enabled" if self.params.show_contours else "disabled"
                    logger.info(f"Contour display {cont}")
                elif key == ord('h'):
                    self.params.use_histogram = not self.params.use_histogram
                    hist = "enabled" if self.params.use_histogram else "disabled"
                    logger.info(f"Histogram equalization {hist}")

                self.frame_count += 1

            logger.info(f"Total frames processed: {self.frame_count}")
            logger.info(f"Unique frames saved: {self.unique_count}")
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
            logger.error("Usage: python 08_Template_Matching_uniqe_edge.py <video_path>")
            return 1

        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return 1

        # Create detector and process video
        detector = UniqueFrameDetector()
        if detector.process_video(video_path):
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
