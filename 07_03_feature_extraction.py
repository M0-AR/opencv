import cv2
import os
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# Function for Multi-Scale Retinex (MSR) enhancement
def MSR(image, sigma_list=[15, 80, 250]):
    retinex = np.zeros_like(image, dtype=np.float32)
    for sigma in sigma_list:
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        retinex += np.log10(image + 1.0) - np.log10(blurred + 1.0)
    retinex /= len(sigma_list)
    return retinex

# Function for Gamma Correction
def gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Function for Histogram Equalization
def histogram_equalization(image):
    return cv2.equalizeHist(image)

# Function for Sharpening
def sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Function to enhance and extract features from a frame
def enhance_and_extract_features(gray_frame, frame_count, output_dir):
    # Apply image enhancement techniques
    enhanced_image_msr = MSR(gray_frame)
    gamma_image = gamma_correction(gray_frame, gamma=1.5)
    equalized_image = histogram_equalization(gray_frame)
    sharpened_image = sharpen(gray_frame)

    # Save enhanced images
    cv2.imwrite(os.path.join(output_dir, f"enhanced_image_msr_{frame_count:04d}.jpg"), enhanced_image_msr)
    cv2.imwrite(os.path.join(output_dir, f"gamma_image_{frame_count:04d}.jpg"), gamma_image)
    cv2.imwrite(os.path.join(output_dir, f"equalized_image_{frame_count:04d}.jpg"), equalized_image)
    cv2.imwrite(os.path.join(output_dir, f"sharpened_image_{frame_count:04d}.jpg"), sharpened_image)

    # Initialize ORB and BRISK detectors
    orb = cv2.ORB_create()
    brisk = cv2.BRISK_create()

    # ORB Feature Detection and Description
    keypoints_orb, descriptors_orb = orb.detectAndCompute(enhanced_image_msr, None)
    image_with_keypoints_orb = cv2.drawKeypoints(enhanced_image_msr, keypoints_orb, None, color=(255, 0, 0))
    cv2.imwrite(os.path.join(output_dir, f"image_with_keypoints_orb_{frame_count:04d}.jpg"), image_with_keypoints_orb)

    # BRISK Feature Detection and Description
    keypoints_brisk, descriptors_brisk = brisk.detectAndCompute(enhanced_image_msr, None)
    image_with_keypoints_brisk = cv2.drawKeypoints(enhanced_image_msr, keypoints_brisk, None, color=(0, 255, 0))
    cv2.imwrite(os.path.join(output_dir, f"image_with_keypoints_brisk_{frame_count:04d}.jpg"), image_with_keypoints_brisk)

    # Local Binary Patterns (LBP) for texture analysis
    lbp = local_binary_pattern(gray_frame, P=8, R=1, method='uniform')
    cv2.imwrite(os.path.join(output_dir, f"lbp_{frame_count:04d}.jpg"), lbp)

    # Gray-Level Co-occurrence Matrix (GLCM) for texture analysis
    glcm = graycomatrix(gray_frame, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Save GLCM properties
    with open(os.path.join(output_dir, f"glcm_properties_{frame_count:04d}.txt"), 'w') as file:
        file.write(f"Contrast: {contrast}\n")
        file.write(f"Dissimilarity: {dissimilarity}\n")
        file.write(f"Homogeneity: {homogeneity}\n")
        file.write(f"Energy: {energy}\n")
        file.write(f"Correlation: {correlation}\n")

# Define paths
video_path = "020146-3173 (35).mp4"  # Replace with the path to your video file
output_dir = "07_03"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize feature detectors and descriptors
fast = cv2.FastFeatureDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create() if hasattr(cv2.xfeatures2d, 'BriefDescriptorExtractor_create') else None
orb = cv2.ORB_create()
brisk = cv2.BRISK_create()

frame_count = 0  # Initialize a counter for naming saved frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = frame[:, 200:]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply FAST + BRIEF
    if brief:
        keypoints_fast = fast.detect(gray_frame, None)
        keypoints_fast, descriptors_brief = brief.compute(gray_frame, keypoints_fast)
        frame_fast_brief = cv2.drawKeypoints(gray_frame, keypoints_fast, None, color=(0, 0, 255))
        cv2.imwrite(os.path.join(output_dir, f"frame_fast_brief_{frame_count:04d}.png"), frame_fast_brief)
    else:
        keypoints_fast = fast.detect(gray_frame, None)
        frame_fast = cv2.drawKeypoints(gray_frame, keypoints_fast, None, color=(0, 0, 255))
        cv2.imwrite(os.path.join(output_dir, f"frame_fast_{frame_count:04d}.png"), frame_fast)

    # Apply ORB
    keypoints_orb, descriptors_orb = orb.detectAndCompute(gray_frame, None)
    frame_orb = cv2.drawKeypoints(gray_frame, keypoints_orb, None, color=(255, 255, 0))
    cv2.imwrite(os.path.join(output_dir, f"frame_orb_{frame_count:04d}.png"), frame_orb)

    # Apply BRISK
    keypoints_brisk, descriptors_brisk = brisk.detectAndCompute(gray_frame, None)
    frame_brisk = cv2.drawKeypoints(gray_frame, keypoints_brisk, None, color=(0, 255, 255))
    cv2.imwrite(os.path.join(output_dir, f"frame_brisk_{frame_count:04d}.png"), frame_brisk)

    # Enhance and extract features
    enhance_and_extract_features(gray_frame, frame_count, output_dir)

    frame_count += 1

# Release the video capture object
cap.release()

print(f"Frames extracted and saved to {output_dir}")
