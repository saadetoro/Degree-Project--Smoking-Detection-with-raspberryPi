import numpy as np
import pickle
import sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data and labels
x = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

data = np.array(x)
labels = np.array(y)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the SVM classifier
classifier = svm.SVC(kernel='linear', C=1.0)

# Train the classifier
classifier.fit(train_data, train_labels)

# Test the classifier
predictions = classifier.predict(test_data)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(test_labels, predictions)

print('Accuracy:', accuracy)
