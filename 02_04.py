"""
Feature Matching:

You can use algorithms like SIFT (Scale-Invariant Feature Transform), SURF (Speeded-Up Robust Features), or ORB (Oriented FAST and Rotated BRIEF) to detect and describe local features in images.
Once you have the features, you can use a feature matching algorithm like FLANN or BFMatcher to find similar features between images.
Structural Similarity Index (SSIM):

This method compares local patterns of pixel intensities that have been normalized for luminance and contrast.
SSIM is good for measuring the similarity of two images perceived by the human visual system.
Mean Squared Error (MSE):

MSE measures the average squared difference between the estimated values and the actual value.
It is simple to compute but can be sensitive to intensity differences across the images.
Hashing Techniques:

Algorithms like pHash (Perceptual Hash) or dHash (Difference Hash) can create a hash value based on the content of the images.
Images are compared by calculating the Hamming distance between their hash values.
Deep Learning:

Deep learning models, particularly Convolutional Neural Networks (CNNs), can be trained to understand the content of an image and can be used to compare the similarity based on high-level features.
Pretrained models like VGG16, ResNet, or Inception can be used to extract feature vectors of images and compare them using cosine similarity or Euclidean distance.
Template Matching:

This method is used to find parts of an image that match a template image. OpenCV provides a function called
matchTemplate for this purpose. It is more suitable for finding exact or very similar objects within the image rather
than comparing two whole images for similarity. Each method has its advantages and is suitable for different
scenarios. The choice of method depends on the specific requirements of your application, such as the type of
similarity you want to detect, the computational resources available, and whether you need to be robust to certain
transformations like scaling, rotation, etc. """
import cv2
import os
import numpy as np
from scipy.spatial.distance import euclidean

# 1. Feature Matching with ORB and BFMatcher
# import cv2
#
# # Load images
# # image1 = cv2.imread('cropped_image_01.jpg', cv2.IMREAD_GRAYSCALE)
# # image2 = cv2.imread('cropped_image_02.jpg', cv2.IMREAD_GRAYSCALE)
#
# image1 = cv2.imread('cropped_image_01.jpg')
# image2 = cv2.imread('cropped_image_02.jpg')
#
# # Initialize ORB detector
# orb = cv2.ORB_create()
#
# # Find the keypoints and descriptors with ORB
# keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
# keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
#
# # Create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
# # Match descriptors
# matches = bf.match(descriptors1, descriptors2)
#
# # Sort them in order of their distance
# matches = sorted(matches, key=lambda x: x.distance)
#
# # Draw first 10 matches
# matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=2)
#
# # Draw first 10 matches and create a resizable window for them
# cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
# cv2.imshow('Matches', matched_image)
# cv2.waitKey(0)

# 2. Structural Similarity Index (SSIM)
from skimage.metrics import structural_similarity as ssim
import cv2

image1 = cv2.imread('cropped_image_01.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('cropped_image_02.jpg', cv2.IMREAD_GRAYSCALE)

# Compute SSIM between two images
score, _ = ssim(image1, image2, full=True)
print(f"SSIM: {score}")

# 3. Mean Squared Error (MSE)
import cv2
import numpy as np

def mse(imageA, imageB):
    # Note: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

image1 = cv2.imread('cropped_image_01.jpg')
image2 = cv2.imread('cropped_image_02.jpg')

# Compute MSE between two images
mse_value = mse(image1, image2)
print(f"MSE: {mse_value}")
