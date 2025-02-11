from PIL import Image
import cv2
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
print(matplotlib.get_backend())
matplotlib.use('TkAgg')  # Use 'Agg' for non-GUI environments

# Load images locally
original = Image.open('extracted_frames_01/frame_00000.jpg')
tampered = Image.open('extracted_frames_01/frame_00006.jpg')

original = Image.open('extracted_frames_01/frame_00006.jpg')
tampered = Image.open('extracted_frames_01/frame_00016.jpg')


# Convert PIL Images to NumPy arrays for easier manipulation with OpenCV
original = np.array(original)
tampered = np.array(tampered)

# Resize images if they are not the same size
if original.shape != tampered.shape:
    tampered = cv2.resize(tampered, (original.shape[1], original.shape[0]))

# Convert images to grayscale
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

# Compute SSIM between two images
(score, diff) = compare_ssim(original_gray, tampered_gray, full=True)
diff = (diff * 255).astype("uint8")

print(f"SSIM: {score:.2f}")

# Threshold the difference image
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Find contours from the threshold image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original and tampered images
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(tampered, (x, y), (x + w, y + h), (0, 255, 0), 2)

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

# Convert PIL Images to NumPy arrays if needed
original = np.array(original)
tampered = np.array(tampered)


# Convert PIL Images to NumPy arrays if needed
original_gray = np.array(original_gray)
tampered_gray = np.array(tampered_gray)

# Create list of images and their titles for display
images = [original, tampered, original_gray, tampered_gray, diff, thresh]
titles = ["Original Image", "Tampered Image", "", "","Difference Image", "Threshold Image"]

# Call the display function
display_images(images, titles)