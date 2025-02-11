"""Image Grid Composition Module.

This module combines multiple images into a grid layout with interactive
controls for customization and preview.

Algorithm Explanation for Beginners:
--------------------------------
We use a step-by-step process to create a grid of images:

1. Image Loading:
   - Read images from a directory
   - Check image sizes and formats
   - Resize if needed for consistency
   - Like organizing photos on a wall

2. Grid Layout:
   - Calculate how many rows and columns we need
   - Like planning a photo collage:
     * If we have 12 photos and want 3 columns
     * We'll need 4 rows to fit them all
   - Make sure spacing is even

3. Image Placement:
   - For each image, find its spot in the grid:
     * Row = image number ÷ columns
     * Column = image number % columns
   - Like putting photos in order on a wall
   - Keep consistent spacing

4. Final Composition:
   - Create a big canvas for all images
   - Copy each image to its spot
   - Add borders and labels if needed
   - Save the final collage

Key Features:
- Automatic grid sizing
- Interactive preview
- Progress feedback
- Error handling
- Resource cleanup
- Performance optimization

Usage:
    python image_grid.py <input_dir> [--cols N] [--border N]

Controls:
    '+/-' - Adjust columns
    'b' - Toggle borders
    'l' - Toggle labels
    's' - Save grid
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
class GridParams:
    """Parameters for grid composition."""
    columns: int = 3
    border_size: int = 2
    border_color: Tuple[int, int, int] = (255, 255, 255)
    target_width: int = 1920
    show_labels: bool = True
    label_height: int = 30
    label_color: Tuple[int, int, int] = (255, 255, 255)
    supported_formats: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')


class ImageGridComposer:
    """A class to handle image grid composition."""

    def __init__(self, output_dir: str = "grid_output"):
        """Initialize the composer.

        Args:
            output_dir: Base directory for output
        """
        self.output_dir = Path(output_dir)
        self.params = GridParams()
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize state
        self.images = []
        self.grid = None
        self.need_update = True

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
            
            logger.info(f"Found {len(image_files)} images")
            
            # Load images
            self.images = []
            for path in image_files:
                image = cv2.imread(str(path))
                if image is None:
                    logger.warning(f"Could not load {path}")
                    continue
                    
                self.images.append((path.stem, image))
            
            if not self.images:
                raise RuntimeError("No valid images loaded")
            
            logger.info(f"Loaded {len(self.images)} images")
            self.need_update = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading images: {str(e)}")
            return False

    def resize_image(
        self,
        image: NDArray,
        target_size: Tuple[int, int]
    ) -> NDArray:
        """Resize image to target size.

        Args:
            image: Input image
            target_size: Target size (width, height)

        Returns:
            Resized image
        """
        try:
            return cv2.resize(image, target_size)
            
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            return image

    def create_grid(self) -> Optional[NDArray]:
        """Create image grid.

        Returns:
            Grid image if successful
        """
        try:
            if not self.images:
                return None
            
            # Calculate grid size
            n_images = len(self.images)
            n_cols = self.params.columns
            n_rows = (n_images - 1) // n_cols + 1
            
            # Get cell size
            _, first_image = self.images[0]
            cell_height, cell_width = first_image.shape[:2]
            
            # Add space for labels
            if self.params.show_labels:
                cell_height += self.params.label_height
            
            # Add borders
            cell_width += 2 * self.params.border_size
            cell_height += 2 * self.params.border_size
            
            # Create grid
            grid_width = n_cols * cell_width
            grid_height = n_rows * cell_height
            grid = np.zeros(
                (grid_height, grid_width, 3),
                dtype=np.uint8
            )
            
            # Fill grid
            for idx, (name, image) in enumerate(self.images):
                # Calculate position
                row = idx // n_cols
                col = idx % n_cols
                
                # Calculate coordinates
                x1 = col * cell_width + self.params.border_size
                x2 = x1 + image.shape[1]
                y1 = row * cell_height + self.params.border_size
                y2 = y1 + image.shape[0]
                
                # Place image
                grid[y1:y2, x1:x2] = image
                
                # Add label
                if self.params.show_labels:
                    y_text = y2 + self.params.label_height // 2
                    cv2.putText(
                        grid,
                        name,
                        (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        self.params.label_color,
                        1
                    )
            
            # Resize if needed
            if grid_width > self.params.target_width:
                aspect_ratio = grid_width / grid_height
                target_height = int(self.params.target_width / aspect_ratio)
                grid = self.resize_image(
                    grid,
                    (self.params.target_width, target_height)
                )
            
            self.grid = grid
            self.need_update = False
            
            return grid
            
        except Exception as e:
            logger.error(f"Error creating grid: {str(e)}")
            return None

    def save_grid(self, path: Optional[str] = None) -> bool:
        """Save grid to file.

        Args:
            path: Optional output path

        Returns:
            True if successful
        """
        try:
            if self.grid is None:
                if not self.create_grid():
                    return False
            
            # Use default path if none provided
            if path is None:
                path = self.output_dir / "grid.jpg"
            else:
                path = Path(path)
            
            # Save grid
            cv2.imwrite(str(path), self.grid)
            logger.info(f"Saved grid to {path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving grid: {str(e)}")
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
        elif key == ord('+'):
            self.params.columns += 1
            self.need_update = True
        elif key == ord('-'):
            if self.params.columns > 1:
                self.params.columns -= 1
                self.need_update = True
        elif key == ord('b'):
            self.params.border_size = \
                0 if self.params.border_size else 2
            self.need_update = True
        elif key == ord('l'):
            self.params.show_labels = not self.params.show_labels
            self.need_update = True
        elif key == ord('s'):
            self.save_grid()
        return True

    def run(self) -> bool:
        """Run interactive preview.

        Returns:
            True if successful
        """
        try:
            while True:
                # Update grid if needed
                if self.need_update:
                    grid = self.create_grid()
                    if grid is None:
                        return False
                    
                    # Show grid
                    cv2.imshow('Grid', grid)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key(key):
                    break
            
            return True
            
        except Exception as e:
            logger.error(f"Error running preview: {str(e)}")
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
            description='Create image grid composition'
        )
        parser.add_argument(
            'input_dir',
            help='Directory containing images'
        )
        parser.add_argument(
            '--cols',
            type=int,
            default=3,
            help='Number of columns'
        )
        parser.add_argument(
            '--border',
            type=int,
            default=2,
            help='Border size'
        )
        
        args = parser.parse_args()
        
        # Create composer
        composer = ImageGridComposer()
        composer.params.columns = args.cols
        composer.params.border_size = args.border
        
        # Load images
        if not composer.load_images(args.input_dir):
            return 1
        
        # Run preview
        if not composer.run():
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
