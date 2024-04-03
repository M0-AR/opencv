"""
A straightforward approach for exact matches (identical frames) won't be very effective in real-world scenarios due to slight movements, lighting changes, and camera noise. Instead, you can use feature matching or scene recognition techniques to identify similar frames. Here's a conceptual outline of how you might approach this:

1. Feature Extraction and Matching
For each frame, extract key features (using algorithms like SIFT, SURF, or ORB) that can robustly describe the visual content of the frame. When you process a new frame, compare its features to the features of previously processed frames. If you find a match (based on a similarity threshold), you can consider the new frame as "already seen."

2. Hashing for Similarity Detection
A simpler approach might involve computing a perceptual hash (like pHash) for each frame. These hashes are designed to be similar for visually similar images, even if there are minor changes in lighting or perspective. By comparing the hash of a new frame to the hashes of previously seen frames, you can decide if a frame has been seen based on hash similarity.

Example Using Feature Matching (Conceptual)
Here's a conceptual Python snippet using OpenCV's feature matching. This example doesn't implement the full solution but gives a starting point:
"""
import cv2
import os
import numpy as np

# Initialize the ORB detector
orb = cv2.ORB_create()

# Placeholder for storing features of seen frames
seen_frames_features = []


# Function to check if frame is similar to already seen frames
def is_frame_seen(new_frame):
    keypoints_new, descriptors_new = orb.detectAndCompute(new_frame, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    for _, desc in seen_frames_features:
        matches = bf.knnMatch(desc, descriptors_new, k=2)

        # Apply ratio test
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        # If enough matches are found, we consider the frame as seen
        if len(good_matches) > 10:  # The threshold here is arbitrary
            return True

    # If no match is found, add this frame's features to the seen frames
    seen_frames_features.append((keypoints_new, descriptors_new))
    return False

# Specify the output directory for saving new frames
output_dir = '05_feature_matching'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Path to your video file
video_path = '020146-3173 (35).mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0  # Initialize a counter for naming saved frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cropped_image = frame[:, 200:]

    # Convert the frame to grayscale for the is_frame_seen check
    frame_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Check if the frame has been seen
    if is_frame_seen(frame_gray):
        print("Frame already seen, skipping...")
        continue

    print("New frame")
    # Process the frame here if it hasn't been seen

    # Construct a filename using the frame count
    frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
    # Save the frame
    cv2.imwrite(frame_filename, frame)
    print(f"Saved {frame_filename}")

    frame_count += 1  # Increment the frame counter

    # Optionally, break the loop if 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release resources
cap.release()
# cv2.destroyAllWindows()