"""
To compare images in chunks and remove similar ones based on the similarity threshold, you can modify your script as follows:

Divide the list of images into chunks of a specific size.
In each chunk, compare the first image to the rest and mark similar images for removal.
After processing each chunk, remove the marked images and proceed to the next chunk.
Repeat this process for all chunks until no further images can be removed.
Here's the modified script with chunk processing:
"""
import cv2
import os
import numpy as np
from scipy.spatial.distance import euclidean


def extract_color_histogram(image, bins=(8, 8, 8)):
    """Extract a 3D color histogram from the HSV color space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def get_image_paths(directory):
    """Get a list of image file paths in a directory."""
    image_files = sorted(f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    return [os.path.join(directory, f) for f in image_files]


def remove_similar_images(directory, chunk_size, similarity_threshold):
    """Remove images that are similar within the next 'chunk_size' images."""
    image_paths = get_image_paths(directory)
    removed_count = 0

    # Iterate over images as reference images
    for i in range(len(image_paths)):
        if i >= len(image_paths) - 1:  # Skip the last image
            break

        reference_path = image_paths[i]
        reference_image = cv2.imread(reference_path)
        if reference_image is None:
            continue  # Skip if the image cannot be loaded
        reference_hist = extract_color_histogram(reference_image)

        # Determine the range for comparison
        compare_range = min(i + 1 + chunk_size, len(image_paths))

        # Track indices to remove
        to_remove_indices = []

        # Compare the reference image to subsequent images
        for j in range(i + 1, compare_range):
            compare_path = image_paths[j]
            current_image = cv2.imread(compare_path)
            if current_image is None:
                continue
            current_hist = extract_color_histogram(current_image)
            distance = euclidean(reference_hist, current_hist)

            # If the images are similar, mark the index for removal
            if distance < similarity_threshold:
                to_remove_indices.append(j)
                os.remove(compare_path)
                removed_count += 1
                print(f"Removed: {compare_path}")

        # Remove indices from image_paths
        for index in sorted(to_remove_indices, reverse=True):
            del image_paths[index]

    return removed_count


# Directory containing the images
directory = 'extracted_frames_01'

# Chunk size and similarity threshold
chunk_size = 30  # Number of images to compare to each reference image
similarity_threshold = 0.7  # Similarity threshold

# Run the function
removed_total = remove_similar_images(directory, chunk_size, similarity_threshold)

print(f"Removed a total of {removed_total} similar images from {directory}.")