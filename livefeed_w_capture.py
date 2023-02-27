import cv2
import time
import datetime
import os
import tensorflow as tf

def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Load the trained model
model = tf.keras.models.load_model("CNN.model")

# Set the time interval between frames (in seconds)
time_interval = 5

# Open the webcam
cap = cv2.VideoCapture(0)
start_time = time.time()

try:
    if not os.path.exists('CNN_test'):
        os.makedirs('CNN_test')
except OSError:
    print('No Such Directory')

while True:
    # Capture a frame
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if ret:
        cv2.imshow('Camera', frame)
        
        # Preprocess the frame
        #frame = cv2.resize(frame, (224,224)) # resize to the input size of the model
        frame = prepare(frame)
        #frame = frame / 255.0 # normalize the pixel values
        #frame = frame[None, 50, 50, 1] # add a batch dimension

        #Predict whether the frame contains a smoking person
        prediction = model.predict(frame)[0][0]

        if prediction == "Smoking":
            #test image on haar cascade here

            #if haar cascade is positive:
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
cv2.destroyAllWindows()#