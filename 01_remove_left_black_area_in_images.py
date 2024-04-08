import cv2
import os

# Directory containing the images
directory = 'extracted_frames'

# Loop through each file in the directory
for filename in os.listdir(directory):
    # Check if the file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Construct the full file path
        file_path = os.path.join(directory, filename)

        # Read the image
        image = cv2.imread(file_path)

        # Check if the image was loaded successfully
        if image is not None:
            # Crop the image from x=200 to the end
            cropped_image = image[:, 200:]

            # Optionally, display the cropped image
            # cv2.imshow('Cropped Image', cropped_image)
            # cv2.waitKey(0)

            # Save the cropped image back to the same file
            cv2.imwrite(file_path, cropped_image)

# Destroy all windows if opened
cv2.destroyAllWindows()
