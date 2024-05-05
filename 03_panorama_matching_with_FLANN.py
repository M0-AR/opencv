import cv2
import numpy as np

# Load images
img1 = cv2.imread('cropped_image_01.jpg')  # queryImage
img2 = cv2.imread('cropped_image_02.jpg')  # trainImage

# Initialize SIFT feature detector and descriptor extractor
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# FLANN parameters and matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Apply Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.9 * n.distance:
        good_matches.append(m)

# Draw matches that passed the ratio test
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)

# Show the matches
cv2.namedWindow('Matches after Ratio Test', cv2.WINDOW_NORMAL)
cv2.imshow('Matches after Ratio Test', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Proceed with homography if there are enough good matches
if len(good_matches) > 10:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute Homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp second image to fit the first image
    height, width, _ = img1.shape
    img2_transformed = cv2.warpPerspective(img2, H, (width, height))

    # Stitch images and show the result
    result = cv2.hconcat([img1, img2_transformed])
    cv2.namedWindow('Stitched Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Stitched Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result
    cv2.imwrite('result.jpg', result)
else:
    print("Not enough good matches - {}/{}".format(len(good_matches), 10))
