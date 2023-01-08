import cv2
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy
import time
import datetime
import os

# Define the model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Flatten the output and add dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the training and test data

train_path = 'archive\data\train'

train_data_generator = ImageDataGenerator(rescale=1./255,
                                         rotation_range=45,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True,
                                         fill_mode='nearest')

test_path = 'archive\data\test\test'

test_data_generator = ImageDataGenerator(rescale=1./255)

# Load the training and test data
train_data = train_data_generator.flow_from_directory(r'C:\Users\User\Documents\School\Degree Project\Degree-Project--Smoking-Detection-with-raspberryPi\archive\data\train',
                                                      target_size=(224,224),
                                                      batch_size=64,
                                                      class_mode='binary')
test_data = test_data_generator.flow_from_directory(r'C:\Users\User\Documents\School\Degree Project\Degree-Project--Smoking-Detection-with-raspberryPi\archive\data\test',
                                                     target_size=(224,224),
                                                     batch_size=1,
                                                     class_mode='binary')

# Train the model
model.fit_generator(train_data,
                    epochs=10,
                    validation_data=test_data)

# Save the model
model.save('smoking_detection_model.h5')

# Load the trained model
model = load_model('smoking_detection_model.h5')

# Set the time interval between frames (in seconds)
time_interval = 5

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
        start_time = current_time

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
