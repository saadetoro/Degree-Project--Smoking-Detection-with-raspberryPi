'''This code loads a pre-trained CNN model (which you will need to train and save beforehand) and 
opens the webcam. It then captures a frame, preprocesses it (resizing and normalizing the pixel values), 
and makes a prediction using the predict() method. The prediction is displayed on the frame using the putText() 
function and the frame is displayed using the imshow() function. The loop continues until the user presses the 
'q' key, at which point the webcam is released and all windows are closed.'''


import cv2
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.models import load_model

# Load the trained model
model = load_model('smoking_detection_model.h5')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        break

    # Preprocess the frame
    frame = cv2.resize(frame, (224,224)) # resize to the input size of the model
    frame = frame / 255.0 # normalize the pixel values
    frame = frame[None, ...] # add a batch dimension

    # Predict whether the frame contains a smoking person
    prediction = model.predict(frame)[0][0]

    # Display the prediction
    if prediction > 0.5:
        cv2.putText(frame, 'Smoking', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
    else:
        cv2.putText(frame, 'Non-smoking', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()