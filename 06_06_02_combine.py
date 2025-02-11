import cv2
import numpy as np
import os

# Define paths
saved_frames_dir = "06_06"  # Directory where frames are saved
output_image_path = "combined_image.jpg"  # Path to save the combined image

# Read all image files from the directory
image_files = sorted([f for f in os.listdir(saved_frames_dir) if f.endswith('.jpg')])

# Check if there are images to process
if not image_files:
    print(f"No images found in {saved_frames_dir}.")
    exit()

# Load the images
saved_frames = [cv2.imread(os.path.join(saved_frames_dir, file)) for file in image_files]

# Check if images are loaded correctly
if any(frame is None for frame in saved_frames):
    print(f"Error loading one or more images from {saved_frames_dir}.")
    exit()

# Combine saved frames into one image (e.g., create a grid of saved frames)
grid_size = (len(saved_frames) // 3 + (1 if len(saved_frames) % 3 != 0 else 0), 3)  # Adjust grid size as needed
frame_height, frame_width = saved_frames[0].shape[:2]
combined_image = np.zeros((grid_size[0] * frame_height, grid_size[1] * frame_width, 3), dtype=np.uint8)

for idx, frame in enumerate(saved_frames):
    row = idx // grid_size[1]
    col = idx % grid_size[1]
    combined_image[row * frame_height:(row + 1) * frame_height, col * frame_width:(col + 1) * frame_width] = frame

# Save the combined image
cv2.imwrite(output_image_path, combined_image)

print(f"Combined image saved as '{output_image_path}'.")
