import cv2
import numpy as np

# Load images
img1 = cv2.imread('cropped_image_01.jpg')  # queryImage
img2 = cv2.imread('cropped_image_02.jpg')  # trainImage

# Initialize BRISK feature detector and descriptor extractor
brisk = cv2.BRISK_create()

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = brisk.detectAndCompute(img1, None)
keypoints2, descriptors2 = brisk.detectAndCompute(img2, None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Perform k-NN matching with k=2 (find the two closest matches for each descriptor)
pairs_of_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply the ratio test
good_matches = [m for m, n in pairs_of_matches if len(pairs_of_matches) > 1 and m.distance < 0.8 * n.distance]

# Sort the good matches based on distance
good_matches = sorted(good_matches, key=lambda x: x.distance)

# Draw the best 25 matches
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches[:25], None, flags=2)

# Convert BGR to RGB for matplotlib display
img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

# Show the matches
cv2.namedWindow('Matches after Ratio Test', cv2.WINDOW_NORMAL)
cv2.imshow('Matches after Ratio Test', img_matches_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

# If there are enough good matches, proceed to find homography and warp image
if len(good_matches) > 10:
    # Extract location of good matches
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

    # Use homography to warp the second image
    height, width, _ = img1.shape
    img2_transformed = cv2.warpPerspective(img2, H, (width, height))

    # Stitch images together
    result = cv2.hconcat([img1, img2_transformed])

    # Save the result
    cv2.imwrite('result.jpg', result)
else:
    print("Not enough good matches - {}/{}".format(len(good_matches), 10))
