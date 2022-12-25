'''This code creates a CNN with four convolutional layers, two max pooling layers, and two dense layers. 
The input shape is (224,224,3), which means the input images are 224x224 pixels and have 3 color channels (RGB).
The output of the model is a single sigmoid activation, which represents the probability of the input image 
containing a smoking person.
You will need to have training data (i.e., images of smoking and non-smoking people) and 
their corresponding labels (1 for smoking and 0 for non-smoking) in order to train and test the model. 
You can use the fit() and evaluate() methods to train and test the model, respectively.'''


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

'''This will split the data into a training set (80% of the data), a validation set (10% of the data), and 
a test set (10% of the data). You can then use the training set to train the model, the validation set 
to tune the model's hyperparameters, and the test set to evaluate the model's performance.'''

# Load the images and labels
X = ... # a numpy array of shape (n_samples, height, width, channels)
y = ... # a numpy array of shape (n_samples,)

# Split the data into a training set, a validation set, and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

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

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))

# Test the model
model.evaluate(x_test, y_test)


model.save('models/smoking_detection_model.h5')