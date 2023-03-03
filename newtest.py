import cv2
import time
import datetime
import os
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
    model = tf.keras.models.load_model("CNN.model")
    
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
            else:
                csd = haar_test(img_path)
                if csd == '0':
                    os.remove(img_path)

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()