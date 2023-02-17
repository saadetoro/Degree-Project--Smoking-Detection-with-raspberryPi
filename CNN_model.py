import cv2
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
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
model.fit(train_data,
                    epochs=10,
                    validation_data=test_data)

# Save the model
model.save('smoking_detection_model.h5')