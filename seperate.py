import cv2
import os

def extract_frames_from_video(video_path, frames_per_image=10, output_folder="extracted_frames"):
    """
    Extracts images from a video file, saving every Nth frame.

    Args:
        video_path (str): The path to the input video file.
        frames_per_image (int): The number of frames to skip before saving the next image.
                                 A value of 1 saves every frame. A value of 10 saves every 10th frame.
        output_folder (str): The name of the folder where extracted images will be saved.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    frame_count = 0
    saved_image_count = 0

    print(f"Starting frame extraction from: {video_path}")
    print(f"Saving every {frames_per_image} frames...")

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If 'ret' is False, it means we have reached the end of the video
        if not ret:
            break

        # Check if the current frame is one we want to save
        if frame_count % frames_per_image == 0:
            # Construct the output image file name
            # Using f-strings for easy formatting, padding with leading zeros for sorting
            image_name = os.path.join(output_folder, f"frame_{saved_image_count:06d}.jpg")

            # Save the frame as a JPEG image
            cv2.imwrite(image_name, frame)
            saved_image_count += 1

            # Print progress
            if saved_image_count % 100 == 0: # Print update every 100 saved images
                print(f"  Saved {saved_image_count} images so far...")

        frame_count += 1

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    print(f"\nFinished extracting frames.")
    print(f"Total frames processed: {frame_count}")
    print(f"Total images saved: {saved_image_count} in '{output_folder}' folder.")

if __name__ == "__main__":
    # --- Example Usage ---
    # IMPORTANT: Replace 'your_video.mp4' with the actual path to your video file.
    # For example: 'C:/Users/YourUser/Videos/my_awesome_video.mp4' or './test_video.avi'
    # If the video is in the same directory as this script, just the filename is enough.

    video_file_path = input("Please enter the full path to your video file (e.g., C:/Videos/my_video.mp4): ")

    # You can change 'frames_to_skip' to save frames at different intervals.
    # 1: saves every frame
    # 30: saves one frame per second if video is 30fps
    frames_to_skip = 50

    # Call the function to start extraction
    extract_frames_from_video(video_file_path, frames_per_image=frames_to_skip)
