# import cv2
# import numpy as np
#
# # Load the image from the file system
# image = cv2.imread('frame_01269.jpg')
#
# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Define a threshold to detect non-black pixels (change this depending on your image)
# threshold = 10
#
# # Threshold the grayscale image to get the non-black regions
# _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
#
# # Find contours from the thresholded image
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # Find the largest contour which will be the non-black area
# largest_contour = max(contours, key=cv2.contourArea)
#
# # Get the bounding box of the largest contour
# x, y, w, h = cv2.boundingRect(largest_contour)
#
# # Crop the image using the dimensions of the bounding box
# cropped_image = image[y:y+h, x:x+w]
#
# # Display the cropped image
# cv2.imshow('Cropped Image', cropped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np

# Read the image from the file system
image = cv2.imread('frame_01313.jpg')

# Get the dimensions of the image
height, width = image.shape[:2]

# Find the center of the image
center_x, center_y = width // 2, height // 2

# Define the radius of the circular region you want to keep
radius = 350 # Change this to your desired radius

# Create a black mask with the same dimensions as the image
mask = np.zeros((height, width), dtype="uint8")

# Draw a white circle in the middle of the mask with the defined radius
cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)

# Create an all black image with the same size as the original image
output = np.zeros_like(image)

# Copy only the region of interest from the original image into the output image
output[mask == 255] = image[mask == 255]

# Optionally, save the result to disk
cv2.imwrite('cropped_circle_02.jpg', output)

# Display the new image
cv2.imshow('Circular Cropped Image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()