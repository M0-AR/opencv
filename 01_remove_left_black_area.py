import cv2

# Read the image
image = cv2.imread('frame_01462.jpg')

# Remove the left side of the image from x=0 to x=20
# This means we keep all rows (height of the image) and the columns from x=20 to the end of the image width
cropped_image = image[:, 200:]

# Display the cropped image
# Create a named window that can be resized
cv2.namedWindow('Cropped Image', cv2.WINDOW_NORMAL)
# Display the cropped image in the resizable window
cv2.imshow('Cropped Image', cropped_image)
# Wait for a key press before closing the window
cv2.waitKey(0)
# Destroy all windows
cv2.destroyAllWindows()

# Optionally, save the cropped image
cv2.imwrite('cropped_image_02.jpg', cropped_image)
