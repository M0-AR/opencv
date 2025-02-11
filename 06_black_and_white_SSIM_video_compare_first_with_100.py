import cv2
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

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Function to read and crop a specific frame from the video
def read_and_crop_frame(video_capture, frame_number):
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video_capture.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number}.")
        return None
    frame = frame[:, 200:]  # Apply cropping
    return frame

# Specify the reference frame index and other frames to compare
reference_frame_index =0
reference_frame_index = 2

# Read specific frames
# frame_indices = [0, 100, 1000]  # Frames you want to compare with frame 0
frame_indices = [0, 578, 579, 580, 1000, 1500, 2000]  # Frames you want to compare with frame 0
frames = [read_and_crop_frame(cap, idx) for idx in frame_indices]

# Ensure all frames were read successfully
if any(frame is None for frame in frames):
    print("Error reading one or more frames.")
    cap.release()
    exit()

# Convert frames to grayscale
gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

# Compare frame 0 with the other frames
for i in range(1, len(frame_indices)):
    (score, diff) = compare_ssim(gray_frames[reference_frame_index], gray_frames[i], full=True)
    diff = (diff * 255).astype("uint8")

    print(f"Frame {reference_frame_index} vs Frame {frame_indices[i]}, SSIM: {score:.2f}")

    # Threshold the difference image
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find contours from the threshold image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the frames
    frame_0_with_contours = frames[reference_frame_index].copy()
    current_frame_with_contours = frames[i].copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame_0_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(current_frame_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display images
    images = [frame_0_with_contours, current_frame_with_contours, gray_frames[reference_frame_index], gray_frames[i], diff, thresh]
    titles = [f"Frame 0", f"Frame {frame_indices[i]}", "Frame 0 Gray", f"Frame {frame_indices[i]} Gray", "Difference Image", "Threshold Image"]
    display_images(images, titles)

# Release the video capture object
cap.release()

print("Video processing completed.")