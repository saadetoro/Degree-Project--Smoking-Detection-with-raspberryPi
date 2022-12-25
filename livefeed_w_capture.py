import cv2
import time
import datetime
import os

# Set the time interval between frames (in seconds)
time_interval = 5

# Open the webcam
cap = cv2.VideoCapture(0)
start_time = time.time()

while True:
    # Capture a frame
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        break

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Save the frame at the specified time interval
    current_time = time.time()
    if current_time - start_time > time_interval:
        # Get the current date and time
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        savepath = r'C:\Users\User\Documents\School\Degree Project\Degree-Project--Smoking-Detection-with-raspberryPi\data\CNN_test'
        os.chdir(savepath)

        # Save the frame to a file with the current date and time in the file name
        cv2.imwrite(f'webcam_{date_time}.jpg', frame)

        # Reset the start time
        start_time = current_time

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()