import datetime
import os
import cv2
import tensorflow as tf
import numpy as np


def cnn_test(image_path):
    IMG_SIZE = 50
    CATEGORIES = ["Smoking", "Not-Smoking"]
    
    # Read the image and resize
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    # Reshape and normalize the image data
    image = np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    
    # Load the saved model
    model = tf.keras.models.load_model("SDM.h5")
    
    # Make a prediction on the preprocessed image
    prediction = model.predict([image])
    
    # Get the predicted label and return it
    predicted_label = CATEGORIES[np.argmax(prediction)]
    return predicted_label

def haar_test(image_path):
    #Load the pre-trained Haar Cascade classifiers
    smoke_classifier = cv2.CascadeClassifier(r'C:\Users\User\Documents\School\Degree Project\Degree-Project--Smoking-Detection-with-raspberryPi\cascade.xml')
    
    #load image
    img = cv2.imread(image_path)
    
    #Convert image to grayscale
    cvGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #test the image with the classifier
    smoke = smoke_classifier.detectMultiScale(cvGray, 1.1, 4, minSize=(24, 24))
    
    #if smoke is detected return 1 otherwise return 0
    if len(smoke) > 0:
        return 1
    else:
        return 0
    

# Create a VideoCapture object to capture video from the default camera
cap = cv2.VideoCapture(0)

# Create a folder to store the frames
folder_path = 'frames'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Set the time interval to capture frames in seconds
interval = 15

# Set the initial time
prev_time = datetime.datetime.now()

# Set the base file name for the frames
base_name = 'MBGLE0001'

# Set the file extension for the frames
file_ext = '.jpg'

while True:
    # Capture a frame from the video feed
    ret, frame = cap.read()

    # Get the current time
    curr_time = datetime.datetime.now()

    # Calculate the time difference
    delta_time = curr_time - prev_time

    # Save the frame if the time interval has passed
    if delta_time.total_seconds() >= interval:
        # Generate a file name with a time stamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{base_name}_{timestamp}{file_ext}"

        # Save the frame to the folder with the generated file name
        cv2.imwrite(os.path.join(folder_path, filename), frame)

        # Update the previous time
        prev_time = curr_time

    # Display the frame
    cv2.imshow('MBGLE0001', frame)

    # Loop through the images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)

            # Call the CNN and Haar classifiers on the image
            cnn_prediction = cnn_test(img_path)
            haar_csd = haar_test(img_path)

            # Remove the image if it's not smoking or if it has low confidence
            if cnn_prediction == "Not-Smoking" or haar_csd == '0':
                os.remove(img_path)


    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
