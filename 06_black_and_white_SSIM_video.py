import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
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

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

frame_count = 0
# Apply cropping to the first frame
prev_frame = prev_frame[:, 200:]


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = frame[:, 200:]

    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between consecutive frames
    (score, diff) = compare_ssim(prev_gray, current_gray, full=True)
    diff = (diff * 255).astype("uint8")

    print(f"Frame {frame_count}, SSIM: {score:.2f}")

    # Threshold the difference image
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find contours from the threshold image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original and current frames
    frame_with_contours = frame.copy()
    prev_frame_with_contours = prev_frame.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(prev_frame_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display images
    images = [prev_frame_with_contours, frame_with_contours, prev_gray, current_gray, diff, thresh]
    titles = [f"Previous Frame {frame_count-1}", f"Current Frame {frame_count}", "", "", "Difference Image", "Threshold Image"]
    display_images(images, titles)

    # Update the previous frame
    prev_frame = frame.copy()
    frame_count += 1

# Release the video capture object
cap.release()

print("Video processing completed.")
