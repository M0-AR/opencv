# https://datahacker.rs/005-how-to-create-a-panorama-image-using-opencv-with-python/
import cv2
import numpy as np

# Load images (Replace 'path_to_image_1.jpg' and 'path_to_image_2.jpg' with your image files)
img1 = cv2.imread('cropped_image_01.jpg')  # queryImage
img2 = cv2.imread('cropped_image_02.jpg')  # trainImage

# img1 = cv2.imread('cropped_image_01.jpg', cv2.IMREAD_GRAYSCALE)  # queryImage
# img2 = cv2.imread('cropped_image_02.jpg', cv2.IMREAD_GRAYSCALE)  # trainImage

# Create resizable windows for images
cv2.namedWindow('Original Image 1', cv2.WINDOW_NORMAL)
cv2.namedWindow('Original Image 2', cv2.WINDOW_NORMAL)

# Show original images in resizable windows
cv2.imshow('Original Image 1', img1)
cv2.imshow('Original Image 2', img2)
cv2.waitKey(0)

# Detect keypoints using ORB detector
# orb = cv2.ORB_create()
# keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
# keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# SIFT feature detector and descriptor extractor
# sift = cv2.SIFT_create()
# keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
# keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# AKAZE feature detector and descriptor extractor
# akaze = cv2.AKAZE_create()
# keypoints1, descriptors1 = akaze.detectAndCompute(img1, None)
# keypoints2, descriptors2 = akaze.detectAndCompute(img2, None)

# BRISK feature detector and descriptor extractor
brisk = cv2.BRISK_create()
keypoints1, descriptors1 = brisk.detectAndCompute(img1, None)
keypoints2, descriptors2 = brisk.detectAndCompute(img2, None)

# Draw keypoints
img1_keypoints = cv2.drawKeypoints(img1, keypoints1, None, color=(0,255,0), flags=0)
img2_keypoints = cv2.drawKeypoints(img2, keypoints2, None, color=(0,255,0), flags=0)

# Create resizable windows for keypoints images
cv2.namedWindow('Keypoints 1', cv2.WINDOW_NORMAL)
cv2.namedWindow('Keypoints 2', cv2.WINDOW_NORMAL)

# Show keypoints in resizable windows
cv2.imshow('Keypoints 1', img1_keypoints)
cv2.imshow('Keypoints 2', img2_keypoints)
cv2.waitKey(0)

# Convert binary descriptors to the correct type if necessary
# if descriptors1.dtype != np.float32:
#     descriptors1 = descriptors1.astype(np.float32)
# if descriptors2.dtype != np.float32:
#     descriptors2 = descriptors2.astype(np.float32)

# Create BFMatcher object and match descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort the matches based on distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches and create a resizable window for them
cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('Matches', img_matches)
cv2.waitKey(0)

# Assuming there are enough matches, find the homography matrix
if len(matches) > 10:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute Homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Use homography to warp the second image
    height, width, _ = img1.shape
    img2_transformed = cv2.warpPerspective(img2, H, (width, height))

    # Create a resizable window for the warped image
    cv2.namedWindow('Warped Image 2', cv2.WINDOW_NORMAL)
    cv2.imshow('Warped Image 2', img2_transformed)
    cv2.waitKey(0)

    # Stitch images together
    result = cv2.hconcat([img1, img2_transformed])

    # Create a resizable window for the stitched image
    cv2.namedWindow('Stitched Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Stitched Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result
    cv2.imwrite('result.jpg', result)
else:
    print("Not enough matches - {}/{}".format(len(matches), 10))
    cv2.destroyAllWindows()









# https://pyimagesearch.com/2016/01/25/real-time-panorama-and-image-stitching-with-opencv/