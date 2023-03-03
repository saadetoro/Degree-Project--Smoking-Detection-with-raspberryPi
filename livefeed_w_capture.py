import cv2
import time
import datetime
import os
import tensorflow as tf

def cnn_test(frame):
    IMG_SIZE = 50
    img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    model = tf.keras.models.load_model("CNN.model")
    prediction = model.predict([new_array])
    prediction = list(prediction[0])
    return "Smoking" if prediction.index(max(prediction)) == 0 else "Not-Smoking"

# Load the trained model
#model = tf.keras.models.load_model("CNN.model")

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
        #frame = prepare(frame)
        #frame = frame / 255.0 # normalize the pixel values
        #frame = frame[None, 50, 50, 1] # add a batch dimension

        #Predict whether the frame contains a smoking person
        #prediction = model.predict(frame)[0][0]

        #if prediction == "Smoking":
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
    #current_time = time.time()
    #if current_time - start_time > time_interval:
        # Get the current date and time
        #now = datetime.datetime.now()
        #date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        #savepath = r'C:\Users\User\Documents\School\Degree Project\Degree-Project--Smoking-Detection-with-raspberryPi\CNN_test'
        #os.chdir(savepath)

        # Save the frame to a file with the current date and time in the file name
        #cv2.imwrite(f'webcam_{date_time}.jpg', frame)

        # Reset the start time
        #start_time = time.time()



import os

# define the path to the folder containing the images
img_folder = r'C:\Users\User\Documents\School\Degree Project\Degree-Project--Smoking-Detection-with-raspberryPi\CNN_test'

# loop through the images in the folder
for filename in os.listdir(img_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(img_folder, filename)
        # call the cnn_test function on the image
        prediction = cnn_test(img_path)
        if prediction == "Not-Smoking":
            os.remove(img_path)