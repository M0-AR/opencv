"""the script to compare each image with the next one, and then repeat this process for a user-defined number of
iterations, you would need to adjust your function to loop through the images, compare each with its immediate
successor, and then repeat this process as many times as specified. Below is the modified script: """
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


def remove_similar_images(directory, similarity_threshold, iterations):
    """Remove images that are similar to the next one, repeated for a number of iterations."""
    image_paths = get_image_paths(directory)
    removed_count = 0

    for iteration in range(iterations):
        i = 0
        # Use a while loop since the list size can change during iteration
        while i < len(image_paths) - 1:
            reference_path = image_paths[i]
            reference_image = cv2.imread(reference_path)
            if reference_image is None:
                i += 1
                continue  # Skip if the image cannot be loaded
            reference_hist = extract_color_histogram(reference_image)

            compare_path = image_paths[i + 1]
            current_image = cv2.imread(compare_path)
            if current_image is None:
                i += 1
                continue
            current_hist = extract_color_histogram(current_image)
            distance = euclidean(reference_hist, current_hist)

            # If the images are similar, remove the second one
            if distance < similarity_threshold:
                os.remove(compare_path)
                print(f"Removed: {compare_path}")
                removed_count += 1
                # Remove the path from the list
                image_paths.pop(i + 1)
            else:
                i += 1  # Move to the next pair

        print(f"Iteration {iteration + 1}/{iterations} complete.")

    return removed_count

# Directory containing the images
directory = 'extracted_frames_01'

# Similarity threshold and iterations
# similarity_threshold = 0.88  # Similarity threshold (35)
similarity_threshold = 0.88  # Similarity threshold
iterations = 5  # Number of times to repeat the process

# Run the function
removed_total = remove_similar_images(directory, similarity_threshold, iterations)

print(f"Removed a total of {removed_total} similar images from {directory}.")