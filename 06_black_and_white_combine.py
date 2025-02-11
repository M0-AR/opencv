import cv2
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images


def stitch_images_manually(images):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Store the final result of the stitching
    last_image = images[0]  # Start with the first image

    for i in range(1, len(images)):
        # Convert images to grayscale
        gray_last = cv2.cvtColor(last_image, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

        # Detect and compute with SIFT
        keypoints_last, descriptors_last = sift.detectAndCompute(gray_last, None)
        keypoints_current, descriptors_current = sift.detectAndCompute(gray_current, None)

        # Match descriptors between images
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        matches = matcher.knnMatch(descriptors_last, descriptors_current, k=2)

        # Apply ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Extract location of good matches
        points_last = np.zeros((len(good_matches), 2), dtype=np.float32)
        points_current = np.zeros((len(good_matches), 2), dtype=np.float32)

        for j, match in enumerate(good_matches):
            points_last[j, :] = keypoints_last[match.queryIdx].pt
            points_current[j, :] = keypoints_current[match.trainIdx].pt

        # Find homography
        H, _ = cv2.findHomography(points_current, points_last, cv2.RANSAC)

        # Use homography to warp images
        height, width, channels = last_image.shape
        current_warped = cv2.warpPerspective(images[i], H, (width, height))

        # Blend images
        last_image = blend_images(last_image, current_warped)

    return last_image


def blend_images(image1, image2):
    # Create a mask of the second image shape, assuming image1 dominates the left side
    mask = np.zeros_like(image2)
    mask[:, :mask.shape[1] // 2, :] = 1

    # Blend the images using seamlessClone (Poisson blending)
    blended = cv2.seamlessClone(image2, image1, mask * 255, (image1.shape[1] // 2, image1.shape[0] // 2),
                                cv2.NORMAL_CLONE)
    return blended


# Load and process images
folder_path = 'extracted_frames_01_01'
images = load_images_from_folder(folder_path)

# Stitch images manually
stitched_image = stitch_images_manually(images)

# Display the stitched image
cv2.imshow('Stitched Image', stitched_image)

# Wait until any key is pressed
cv2.waitKey(0)

# Save the stitched image
cv2.imwrite('stitched_image.jpg', stitched_image)

# Destroy all windows to free up resources
cv2.destroyAllWindows()