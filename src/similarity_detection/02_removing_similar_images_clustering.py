"""
Image Clustering and Visualization Module.

This module clusters similar images using color features and machine learning,
providing an interactive visualization of image relationships.

Algorithm Explanation for Beginners:
--------------------------------
We use a step-by-step process to group similar images:

1. Feature Extraction:
   - Convert each image to HSV color space
     (better for color comparison than RGB)
   - Create color histogram (like a color fingerprint)
   - This tells us what colors appear most in the image

2. Dimensionality Reduction (PCA):
   - Our color fingerprint has many numbers (dimensions)
   - PCA helps us simplify this to 2 numbers
   - Like making a map of images where similar ones are close
   - This makes it easier to visualize and cluster

3. Clustering (K-Means):
   - K-Means is like playing a game:
     * Place K random points on our map
     * Assign each image to nearest point
     * Move points to center of their assigned images
     * Repeat until points stop moving much
   - Images in same cluster are similar

4. Interactive Visualization:
   - Show images on a 2D plot
   - Color-code by cluster
   - Click points to view images
   - Adjust parameters in real-time

Key Features:
- Interactive cluster visualization
- Real-time parameter adjustment
- Image preview on click
- Progress tracking
- Detailed logging
- Safe image handling

Usage:
    python clustering.py <directory> [--clusters N] [--bins N]

Controls:
    Click - View image
    '+' - Increase clusters
    '-' - Decrease clusters
    'r' - Rerun clustering
    's' - Save results
    'q' - Quit
"""

from dataclasses import dataclass
import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PIL import Image, ImageTk


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ClusterParams:
    """Parameters for image clustering."""
    n_clusters: int = 5
    hist_bins: Tuple[int, int, int] = (8, 8, 8)
    pca_components: int = 2
    supported_formats: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')
    preview_size: Tuple[int, int] = (300, 300)
    cluster_step: int = 1


class ImageClusterer:
    """A class to handle image clustering and visualization."""

    def __init__(self, output_dir: str = "clustered_images"):
        """Initialize the clusterer.

        Args:
            output_dir: Base directory for output
        """
        self.output_dir = Path(output_dir)
        self.params = ClusterParams()
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Image Clusters")
        
        # Create main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create preview frame
        self.preview_frame = tk.Frame(self.main_frame)
        self.preview_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create preview label
        self.preview_label = tk.Label(self.preview_frame)
        self.preview_label.pack(side=tk.LEFT)
        
        # Create info label
        self.info_label = tk.Label(
            self.preview_frame,
            text="Click a point to view image\n'+'/'-' to adjust clusters\n'r' to rerun\n's' to save\n'q' to quit",
            justify=tk.LEFT
        )
        self.info_label.pack(side=tk.RIGHT)
        
        # Initialize state
        self.features = None
        self.reduced_features = None
        self.image_paths = None
        self.kmeans = None
        self.labels = None
        self.scatter = None

    def extract_color_histogram(self, image: NDArray) -> Optional[NDArray]:
        """Extract color histogram from an image.

        Args:
            image: Input image

        Returns:
            Flattened normalized histogram
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate histogram
            hist = cv2.calcHist(
                [hsv],
                [0, 1, 2],
                None,
                self.params.hist_bins,
                [0, 256, 0, 256, 0, 256]
            )
            
            # Normalize
            cv2.normalize(hist, hist)
            
            return hist.flatten()
            
        except Exception as e:
            logger.error(f"Error extracting histogram: {str(e)}")
            return None

    def load_image(self, path: str) -> Optional[NDArray]:
        """Load an image from file.

        Args:
            path: Path to image file

        Returns:
            Loaded image
        """
        try:
            image = cv2.imread(path)
            if image is None:
                raise RuntimeError(f"Could not read image: {path}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None

    def get_image_paths(self, directory: str) -> List[Path]:
        """Get sorted list of image paths.

        Args:
            directory: Directory to search

        Returns:
            List of image paths
        """
        directory = Path(directory)
        
        # Get all image files
        image_paths = []
        for ext in self.params.supported_formats:
            image_paths.extend(directory.glob(f"*{ext}"))
            
        return sorted(image_paths)

    def extract_features(self, directory: str) -> bool:
        """Extract features from all images.

        Args:
            directory: Directory containing images

        Returns:
            True if successful
        """
        try:
            # Get image paths
            self.image_paths = self.get_image_paths(directory)
            if not self.image_paths:
                logger.error(f"No images found in {directory}")
                return False
                
            logger.info(f"Found {len(self.image_paths)} images")
            
            # Extract features
            features = []
            for path in self.image_paths:
                image = self.load_image(str(path))
                if image is None:
                    continue
                    
                hist = self.extract_color_histogram(image)
                if hist is None:
                    continue
                    
                features.append(hist)
            
            if not features:
                logger.error("No features extracted")
                return False
                
            self.features = np.array(features)
            logger.info(f"Extracted features from {len(features)} images")
            
            return True
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return False

    def reduce_dimensions(self) -> bool:
        """Reduce feature dimensions using PCA.

        Returns:
            True if successful
        """
        try:
            if self.features is None:
                logger.error("No features to reduce")
                return False
                
            # Apply PCA
            pca = PCA(n_components=self.params.pca_components)
            self.reduced_features = pca.fit_transform(self.features)
            
            logger.info(
                f"Reduced dimensions from {self.features.shape[1]} "
                f"to {self.reduced_features.shape[1]}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error reducing dimensions: {str(e)}")
            return False

    def cluster_features(self) -> bool:
        """Cluster reduced features using K-Means.

        Returns:
            True if successful
        """
        try:
            if self.reduced_features is None:
                logger.error("No reduced features to cluster")
                return False
                
            # Apply K-Means
            self.kmeans = KMeans(
                n_clusters=self.params.n_clusters,
                random_state=42
            )
            self.labels = self.kmeans.fit_predict(self.reduced_features)
            
            logger.info(f"Created {self.params.n_clusters} clusters")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clustering features: {str(e)}")
            return False

    def show_preview(self, image_path: Path) -> None:
        """Show image preview.

        Args:
            image_path: Path to image to preview
        """
        try:
            # Load and resize image
            image = Image.open(image_path)
            image.thumbnail(self.params.preview_size)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update preview
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
            
        except Exception as e:
            logger.error(f"Error showing preview: {str(e)}")

    def on_click(self, event) -> None:
        """Handle click event.

        Args:
            event: Click event
        """
        if event.inaxes != self.ax:
            return
            
        # Find closest point
        distances = np.sqrt(
            (self.reduced_features[:, 0] - event.xdata) ** 2 +
            (self.reduced_features[:, 1] - event.ydata) ** 2
        )
        idx = np.argmin(distances)
        
        # Show preview
        self.show_preview(self.image_paths[idx])

    def update_plot(self) -> None:
        """Update cluster plot."""
        try:
            # Clear plot
            self.ax.clear()
            
            # Plot clusters
            self.scatter = self.ax.scatter(
                self.reduced_features[:, 0],
                self.reduced_features[:, 1],
                c=self.labels,
                cmap='tab10'
            )
            
            # Set labels
            self.ax.set_xlabel('PCA Feature 1')
            self.ax.set_ylabel('PCA Feature 2')
            self.ax.set_title(
                f'Image Clusters (n_clusters={self.params.n_clusters})'
            )
            
            # Update canvas
            self.canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating plot: {str(e)}")

    def save_results(self) -> None:
        """Save clustering results."""
        try:
            # Create cluster directories
            for i in range(self.params.n_clusters):
                cluster_dir = self.output_dir / f"cluster_{i}"
                cluster_dir.mkdir(exist_ok=True)
                
                # Copy images to cluster directories
                for path, label in zip(self.image_paths, self.labels):
                    if label == i:
                        dest = cluster_dir / path.name
                        dest.write_bytes(path.read_bytes())
            
            logger.info(f"Saved results to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def on_key(self, event) -> None:
        """Handle key event.

        Args:
            event: Key event
        """
        if event.key == '+':
            self.params.n_clusters += self.params.cluster_step
            self.cluster_features()
            self.update_plot()
        elif event.key == '-':
            if self.params.n_clusters > 2:
                self.params.n_clusters -= self.params.cluster_step
                self.cluster_features()
                self.update_plot()
        elif event.key == 'r':
            self.cluster_features()
            self.update_plot()
        elif event.key == 's':
            self.save_results()
        elif event.key == 'q':
            plt.close('all')
            self.root.quit()

    def process_directory(self, directory: str) -> bool:
        """Process directory to cluster images.

        Args:
            directory: Directory containing images

        Returns:
            True if successful
        """
        try:
            # Extract features
            if not self.extract_features(directory):
                return False
                
            # Reduce dimensions
            if not self.reduce_dimensions():
                return False
                
            # Cluster features
            if not self.cluster_features():
                return False
                
            # Set up plot
            self.update_plot()
            
            # Connect events
            self.canvas.mpl_connect('button_press_event', self.on_click)
            self.canvas.mpl_connect('key_press_event', self.on_key)
            
            # Start GUI
            self.root.mainloop()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing directory: {str(e)}")
            return False
            
        finally:
            plt.close('all')


def main() -> int:
    """Main function.

    Returns:
        0 for success, 1 for failure
    """
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(
            description='Cluster similar images and visualize relationships'
        )
        parser.add_argument(
            'directory',
            help='Directory containing images'
        )
        parser.add_argument(
            '--clusters',
            type=int,
            default=5,
            help='Number of clusters'
        )
        parser.add_argument(
            '--bins',
            type=int,
            default=8,
            help='Number of histogram bins per channel'
        )
        
        args = parser.parse_args()
        
        # Create clusterer
        clusterer = ImageClusterer()
        clusterer.params.n_clusters = args.clusters
        clusterer.params.hist_bins = (args.bins,) * 3
        
        # Process directory
        if clusterer.process_directory(args.directory):
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
