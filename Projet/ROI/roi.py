import cv2

# Open a video file or capture device (0 is the default camera)
cap = cv2.VideoCapture("../Videos/pedestrians.avi")

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the video.")
    exit()

# Initialize a variable to keep track of whether the video is paused or not
paused = False

while True:
    if not paused:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the video has ended
        if not ret:
            print("End of video.")
            break

        # Display the frame
        cv2.imshow('Video', frame)

    # Wait for a key event
    key = cv2.waitKey(30)

    # If the 'p' key is pressed, toggle the pause state
    if key == ord('p'):
        paused = not paused

    # If the 'q' key is pressed, exit the loop
    elif key == ord('q'):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
