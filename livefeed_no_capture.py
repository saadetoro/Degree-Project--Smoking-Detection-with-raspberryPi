import numpy as np
import cv2 as cv
import os
from datetime import datetime


livefeed = cv.VideoCapture(0)

if not livefeed.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = livefeed.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv.imshow('video', frame)

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the livefeedture
livefeed.release()
cv.destroyAllWindows()