"""Black Area Removal Module.

This module provides functionality to remove black areas from images by cropping.
It processes all images in a specified directory, removing a fixed number of
pixels from the left side of each image where black areas typically appear.

The module supports various image formats (PNG, JPG, JPEG) and includes
error handling, logging, and progress tracking.

Date: 2025-02-11
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CropConfig:
    """Configuration for image cropping."""
    crop_start: int = 200  # Number of pixels to crop from left
    supported_formats: Set[str] = None
    backup_suffix: str = '_original'
    create_backup: bool = True

    def __post_init__(self):
        """Initialize default values that can't be set as default parameters."""
        if self.supported_formats is None:
            self.supported_formats = {'.png', '.jpg', '.jpeg'}


class ImageProcessor:
    """Class for processing and cropping images."""

    def __init__(self, config: CropConfig):
        """Initialize image processor.

        Args:
            config: Configuration for image processing
        """
        self.config = config

    def is_supported_format(self, filename: str) -> bool:
        """Check if the file format is supported.

        Args:
            filename: Name of the file to check

        Returns:
            True if the file format is supported, False otherwise
        """
        return any(
            filename.lower().endswith(ext)
            for ext in self.config.supported_formats
        )

    def create_file_backup(self, file_path: Path) -> bool:
        """Create a backup of the original file.

        Args:
            file_path: Path to the file to backup

        Returns:
            True if backup was successful, False otherwise
        """
        try:
            if not self.config.create_backup:
                return True

            backup_path = file_path.with_name(
                f"{file_path.stem}{self.config.backup_suffix}{file_path.suffix}"
            )
            if not backup_path.exists():
                backup_path.write_bytes(file_path.read_bytes())
                logger.debug(f"Created backup: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create backup for {file_path}: {str(e)}")
            return False

    def validate_image(
        self,
        image: Optional[NDArray],
        file_path: Path
    ) -> Tuple[bool, str]:
        """Validate the loaded image.

        Args:
            image: Loaded image array
            file_path: Path to the image file

        Returns:
            Tuple of (is_valid, error_message)
        """
        if image is None:
            return False, f"Failed to load image: {file_path}"

        if image.size == 0:
            return False, f"Empty image: {file_path}"

        if len(image.shape) != 3:
            return False, f"Invalid image format (not RGB/BGR): {file_path}"

        if image.shape[1] <= self.config.crop_start:
            return False, f"Image width ({image.shape[1]}) <= crop_start ({self.config.crop_start})"

        return True, ""

    def crop_image(self, image: NDArray) -> NDArray:
        """Crop the image to remove black area.

        Args:
            image: Input image to crop

        Returns:
            Cropped image
        """
        return image[:, self.config.crop_start:]

    def process_image(self, file_path: Path) -> bool:
        """Process a single image file.

        Args:
            file_path: Path to the image file

        Returns:
            True if processing was successful, False otherwise
        """
        try:
            # Create backup
            if not self.create_file_backup(file_path):
                return False

            # Read image
            image = cv2.imread(str(file_path))
            is_valid, error_msg = self.validate_image(image, file_path)
            if not is_valid:
                logger.error(error_msg)
                return False

            # Crop image
            cropped_image = self.crop_image(image)

            # Save cropped image
            cv2.imwrite(str(file_path), cropped_image)
            logger.debug(f"Processed: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return False


def process_directory(
    directory_path: str,
    config: Optional[CropConfig] = None
) -> Tuple[int, int]:
    """Process all images in a directory.

    Args:
        directory_path: Path to directory containing images
        config: Optional configuration, uses defaults if not provided

    Returns:
        Tuple of (number of successful operations, total number of images)

    Raises:
        FileNotFoundError: If directory does not exist
        PermissionError: If directory is not accessible
    """
    try:
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        # Initialize configuration
        if config is None:
            config = CropConfig()

        # Initialize processor
        processor = ImageProcessor(config)

        # Process images
        success_count = 0
        total_count = 0
        
        image_files = [
            f for f in directory.iterdir()
            if f.is_file() and processor.is_supported_format(f.name)
        ]
        total_count = len(image_files)

        if total_count == 0:
            logger.warning(f"No supported images found in {directory}")
            return 0, 0

        logger.info(f"Found {total_count} images to process")

        for i, file_path in enumerate(image_files, 1):
            if processor.process_image(file_path):
                success_count += 1

            if i % 10 == 0:
                progress = (i / total_count) * 100
                logger.info(f"Progress: {progress:.1f}% ({i}/{total_count})")

        logger.info(
            f"Processing complete. "
            f"Successfully processed {success_count} out of {total_count} images"
        )
        return success_count, total_count

    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")
        raise


def main() -> int:
    """Main function to run black area removal.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Process command line arguments
        if len(sys.argv) != 2:
            logger.error("Usage: python black_area_removal.py <directory_path>")
            return 1

        directory_path = sys.argv[1]
        
        # Configure processing
        config = CropConfig(
            crop_start=200,
            create_backup=True
        )

        # Process directory
        success_count, total_count = process_directory(directory_path, config)
        
        # Return success only if all images were processed
        return 0 if success_count == total_count else 1

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
