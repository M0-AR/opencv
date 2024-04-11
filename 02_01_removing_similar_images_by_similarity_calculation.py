"""
Here's what this script does:

Processes the images and checks the similarity between consecutive images based on the color histogram's Euclidean
distance. If the similarity score (distance) is below a defined threshold (similarity_threshold), the next image is
marked to be skipped. All images that are not marked to be skipped are copied to a new directory, effectively
reducing the number of images by removing the most similar consecutive images. Images are never removed from the
original directory; they're only copied if they meet the criteria. Please adjust the similarity_threshold according
to your needs. A lower threshold will result in fewer images being considered similar (and thus fewer being removed),
while a higher threshold will mark more images as similar.

The reduced_frames directory will contain the non-similar images. Ensure that the path to directory is correct and
that new_directory does not already contain any files that might be overwritten
"""
import cv2
import os
import numpy as np
from scipy.spatial.distance import euclidean
import shutil

def extract_color_histogram(image, bins=(8, 8, 8)):
    """Extract a 3D color histogram from the HSV color space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Directory containing the images
directory = 'extracted_frames'

# List of image paths
image_paths = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

similarity_threshold = 0.2  # This means images with scores lower than this are very similar

# Directory to copy the non-similar images to
new_directory = 'reduced_frames'
os.makedirs(new_directory, exist_ok=True)

# List to keep track of images to skip
images_to_skip = set()

# Iterate through the list of image paths
for i, path in enumerate(image_paths):
    # Skip the image if it's marked as similar to another one
    if path in images_to_skip:
        continue

    current_image = cv2.imread(path)
    current_hist = extract_color_histogram(current_image)

    # Only compare with the next image to prevent double checking
    if i < len(image_paths) - 1:
        next_image = cv2.imread(image_paths[i + 1])
        next_hist = extract_color_histogram(next_image)
        distance = euclidean(current_hist, next_hist)

        # If the images are similar, skip the next image
        if distance < similarity_threshold:
            images_to_skip.add(image_paths[i + 1])

    # Copy the current image to the new directory
    shutil.copy(path, os.path.join(new_directory, os.path.basename(path)))

print(f"Images copied to {new_directory}, with most similar images removed.")