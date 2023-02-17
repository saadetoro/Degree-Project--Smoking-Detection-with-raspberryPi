import cv2
import time
import datetime
import os
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.models import load_model

# Load the trained model
#model = load_model('smoking_detection_model.h5')

# Set the time interval between frames (in seconds)
time_interval = 5

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame
    ret, frame = cap.read()
    start_time = time.time()

    # Check if the frame was successfully captured
    if not ret:
        break

    # Preprocess the frame
    #frame = cv2.resize(frame, (224,224)) # resize to the input size of the model
    #frame = frame / 255.0 # normalize the pixel values
    #frame = frame[None, ...] # add a batch dimension

    # Predict whether the frame contains a smoking person
    #prediction = model.predict(frame)[0][0]

    # Display the prediction
    #if prediction > 0.5:
        #cv2.putText(frame, 'Smoking', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
    #else:
        #cv2.putText(frame, 'Non-smoking', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
    
    # Display the frame
    cv2.imshow('Camera', frame)

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Save the frame at the specified time interval
    current_time = time.time()
    if current_time - start_time > time_interval:
        # Get the current date and time
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        savepath = r'C:\Users\User\Documents\School\Degree Project\Degree-Project--Smoking-Detection-with-raspberryPi\CNN_test'
        os.chdir(savepath)

        # Save the frame to a file with the current date and time in the file name
        cv2.imwrite(f'webcam_{date_time}.jpg', frame)

        # Reset the start time
        start_time = time.time()

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()