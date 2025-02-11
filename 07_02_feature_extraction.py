import cv2
import os

# Define the path to the video file
video_path = "020146-3173 (35).mp4"
output_dir = "07_02"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize feature detectors and descriptors
# SIFT and SURF are patented and not included in pre-built OpenCV
# Using only freely available algorithms

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

    # Apply FAST + BRIEF
    if brief:
        keypoints_fast = fast.detect(frame, None)
        keypoints_fast, descriptors_brief = brief.compute(frame, keypoints_fast)
        frame_fast_brief = cv2.drawKeypoints(frame, keypoints_fast, None, color=(0, 0, 255))
        cv2.imwrite(os.path.join(output_dir, f"frame_fast_brief_{frame_count:04d}.png"), frame_fast_brief)
    else:
        keypoints_fast = fast.detect(frame, None)
        frame_fast = cv2.drawKeypoints(frame, keypoints_fast, None, color=(0, 0, 255))
        cv2.imwrite(os.path.join(output_dir, f"frame_fast_{frame_count:04d}.png"), frame_fast)

    # Apply ORB
    keypoints_orb, descriptors_orb = orb.detectAndCompute(frame, None)
    frame_orb = cv2.drawKeypoints(frame, keypoints_orb, None, color=(255, 255, 0))
    cv2.imwrite(os.path.join(output_dir, f"frame_orb_{frame_count:04d}.png"), frame_orb)

    # Apply BRISK
    keypoints_brisk, descriptors_brisk = brisk.detectAndCompute(frame, None)
    frame_brisk = cv2.drawKeypoints(frame, keypoints_brisk, None, color=(0, 255, 255))
    cv2.imwrite(os.path.join(output_dir, f"frame_brisk_{frame_count:04d}.png"), frame_brisk)

    frame_count += 1

# Release the video capture object
cap.release()

print(f"Frames extracted and saved to {output_dir}")
