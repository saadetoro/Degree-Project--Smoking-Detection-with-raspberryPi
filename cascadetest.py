# HAARCascade Smoke Detection.
import cv2
import datetime

#Define classifier
smoke_classifier = cv2.CascadeClassifier(r'C:\Users\User\Documents\School\Degree Project\Degree-Project--Smoking-Detection-with-raspberryPi\cascade.xml')

# Load input image
# Iterate over images in the folder.
image = cv2.imread('wv5.jpg')

# Convert image to grayscale
cvGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


smoke = smoke_classifier.detectMultiScale(cvGray, 1.1, 4, minSize=(24, 24))

# Iterate over the image
for (x, y, w, h) in smoke:
    # Identify objects with a rectangle frame.
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)



cv2.imshow('', image)
cv2.waitKey(0)
#If image detects smoke, send to monitor center
#Otherwise mark as failed detection
#Include timestamp


