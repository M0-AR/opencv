"""
To create clustering between images in a directory and display the graph, you would typically follow these steps:

Feature Extraction: Extract features from each image. This could be color histograms, SIFT, SURF, or deep learning-based features.
Feature Normalization: Normalize these features to ensure that they are on a comparable scale.
Clustering Algorithm: Apply a clustering algorithm like K-Means, Hierarchical clustering, or DBSCAN to these features.
Visualization: Visualize the clusters, for instance, using a scatter plot if you have reduced the features to two dimensions with PCA or t-SNE.
Here's a simplified example using color histograms and K-Means clustering:
"""

import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def extract_color_histogram(image, bins=(8, 8, 8)):
    """Extract a 3D color histogram from the HSV color space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Directory containing the images
directory = 'extracted_frames'

# List to hold feature vectors
features = []
image_paths = []

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(directory, filename)
        image = cv2.imread(file_path)
        if image is not None:
            hist = extract_color_histogram(image)
            features.append(hist)
            image_paths.append(file_path)

# Convert list of features to numpy array
features = np.array(features)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(reduced_features)
labels = kmeans.labels_

# Plot the clusters
for i in range(len(reduced_features)):
    plt.scatter(reduced_features[i, 0], reduced_features[i, 1], label=f'Cluster {labels[i]}')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.title('Image Clusters')
plt.legend()
plt.show()
