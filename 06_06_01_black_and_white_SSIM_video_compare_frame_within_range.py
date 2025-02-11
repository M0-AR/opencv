import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib

print(matplotlib.get_backend())
matplotlib.use('TkAgg')  # Use 'Agg' for non-GUI environments

def display_images(images, titles, cmap=None):
    """
    Display a list of images in a grid with 2 columns.
    :param images: List of image arrays
    :param titles: List of titles for the images
    :param cmap: Color map for displaying images
    """
    assert len(images) == len(titles), "Each image should have a corresponding title"

    # Set up a subplot grid with 2 columns (number of images / 2 rows)
    n = len(images)
    rows = n // 2 + n % 2
    fig, axs = plt.subplots(rows, 2, figsize=(10, 5 * rows))

    for i, image in enumerate(images):
        ax = axs[i // 2, i % 2] if n > 2 else axs[i % 2]
        ax.imshow(image, cmap=cmap if cmap else 'gray')
        ax.set_title(titles[i])
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Define paths
video_path = "020146-3173 (35).mp4"  # Replace with the path to your video file
video_path = "130257-0317.mp4"
output_folder = "06_06_130"  # Directory to save the frames

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)



# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

# Apply cropping to the first frame
prev_frame = prev_frame[:, 200:]

# Variables
frame_count = 0
saved_frames = [prev_frame]
ssim_threshold = 0.8  # SSIM threshold for retaining frames

# Save the first frame
cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count:04d}.jpg"), prev_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply cropping to the current frame
    frame = frame[:, 200:]

    # Convert frames to grayscale
    last_saved_frame = saved_frames[-1]
    prev_gray = cv2.cvtColor(last_saved_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the last saved frame and the current frame
    (score, diff) = compare_ssim(prev_gray, current_gray, full=True)
    diff = (diff * 255).astype("uint8")

    print(f"Last saved frame vs Frame {frame_count}, SSIM: {score:.2f}")

    if score < ssim_threshold:
        saved_frames.append(frame.copy())

        # Save the current frame
        cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count:04d}.jpg"), frame)

        # Threshold the difference image
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Find contours from the threshold image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original and current frames
        frame_with_contours = frame.copy()
        prev_frame_with_contours = last_saved_frame.copy()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(prev_frame_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display images
        images = [prev_frame_with_contours, frame_with_contours, prev_gray, current_gray, diff, thresh]
        titles = ["Last Saved Frame", f"Current Frame {frame_count}", "Last Saved Frame Gray", "Current Frame Gray", "Difference Image", "Threshold Image"]
        # display_images(images, titles)

    frame_count += 1

# Release the video capture object
cap.release()

print(f"Video processing completed. {len(saved_frames)} frames saved.")

# Combine saved frames into one image (e.g., create a grid of saved frames)
grid_size = (len(saved_frames) // 3 + 1, 3)  # Adjust grid size as needed
frame_height, frame_width = saved_frames[0].shape[:2]
combined_image = np.zeros((grid_size[0] * frame_height, grid_size[1] * frame_width, 3), dtype=np.uint8)

for idx, frame in enumerate(saved_frames):
    row = idx // grid_size[1]
    col = idx % grid_size[1]
    combined_image[row * frame_height:(row + 1) * frame_height, col * frame_width:(col + 1) * frame_width] = frame

# Save the combined image
cv2.imwrite('combined_image.jpg', combined_image)

print("Combined image saved as 'combined_image.jpg'.")
