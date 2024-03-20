import cv2
import os

# Function to extract frames
def extract_frames(video_path, output_folder):
    # Make sure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    video = cv2.VideoCapture(video_path)

    # Initialize frame count
    count = 0

    # Loop until there are frames left
    while True:
        success, frame = video.read()
        if not success:
            break

        # Save frame as JPEG file
        cv2.imwrite(os.path.join(output_folder, f"frame_{count:05}.jpg"), frame)
        count += 1

    # When everything done, release the video capture object
    video.release()
    cv2.destroyAllWindows()

    print(f"Extracted {count} frames from the video and saved to '{output_folder}'")


# Path to the video file
# video_path = 'C:\\Users\\md\\Desktop\\_\\a.mp4'  # Replace with your video file path
video_path = '020146-3173 (35).mp4'  # Replace with your video file path

# Folder where images will be saved
output_folder = 'extracted_frames'  # Replace with your desired output folder path

# Extract frames from video
extract_frames(video_path, output_folder)
