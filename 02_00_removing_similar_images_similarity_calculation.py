"""
This script will do the following:

Create a sorted list of image paths in the given directory.
For each image, it calculates its color histogram.
It then compares this histogram to the histogram of the previous and next images in the list, using the Euclidean distance as the similarity score.
The results are stored in the similarity_scores dictionary, which maps each image path to its similarity scores with the adjacent images.
This approach assumes that the images are in a meaningful order when sorted alphabetically. If the order is different (e.g., chronological), make sure the filenames allow for correct sorting.
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

# Directory containing the images
directory = 'extracted_frames'

# List of image paths
image_paths = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Dictionary to hold the similarity scores
similarity_scores = {}

# Iterate through the list of image paths
for i, path in enumerate(image_paths):
    current_image = cv2.imread(path)
    current_hist = extract_color_histogram(current_image)

    similarity_scores[path] = []

    # Compare with the previous image
    if i > 0:
        prev_image = cv2.imread(image_paths[i - 1])
        prev_hist = extract_color_histogram(prev_image)
        prev_distance = euclidean(current_hist, prev_hist)
        similarity_scores[path].append((image_paths[i - 1], prev_distance))

    # Compare with the next image
    if i < len(image_paths) - 1:
        next_image = cv2.imread(image_paths[i + 1])
        next_hist = extract_color_histogram(next_image)
        next_distance = euclidean(current_hist, next_hist)
        similarity_scores[path].append((image_paths[i + 1], next_distance))

# Write the similarity scores to a file
output_file = 'similarity_scores.txt'
with open(output_file, 'w') as file:
    for current_path, scores in similarity_scores.items():
        file.write(f"Current Image: {current_path}\n")
        for compare_path, score in scores:
            file.write(f"    Compared to {compare_path}, Score: {score}\n")
        file.write("\n")

print(f"Similarity scores written to {output_file}")