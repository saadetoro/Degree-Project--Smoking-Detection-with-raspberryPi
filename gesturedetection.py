import numpy as np
import cv2 as cv
import os
from datetime import datetime


livefeed = cv.VideoCapture(0)

if not livefeed.isOpened():
    print("Cannot open camera")
    exit()

savepath = r'C:\Users\User\Documents\School\Degree Project\Degree-Project--Smoking-Detection-with-raspberryPi\data\CNN_test'
os.chdir(savepath)

n = 0

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

    if n == 50:
        n = 0
        cv.imwrite('Mazda5510_' + str(datetime.now()) + '.jpg', frame)
    n += 1

# When everything done, release the livefeedture
livefeed.release()
cv.destroyAllWindows()


'''def save_frame_camera_cycle(device_num, dir_path, basename, cycle, ext='jpg', delay=1, window_name='frame'):
    livefeed = cv.VideoCapture(device_num)

    if not livefeed.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    n = 0
    while True:
        ret, frame = livefeed.read()
        cv.imshow(window_name, frame)
        if cv.waitKey(delay) & 0xFF == ord('q'):
            break
        if n == cycle:
            n = 0
            ct = datetime.now()
            cv.imwrite('{}_{}.{}'.format(base_path, ct.strftime('%Y/%m/%d_%H:%M:%S'), ext), frame)
        n += 1

    cv.destroyWindow(window_name)


save_frame_camera_cycle(0, 'data/temp', 'camera_capture_cycle', 300)



''' 
# set path in which you want to save images
path = r'C:\Users\vishal\Documents\Bandicam'
 
# changing directory to given path
os.chdir(path)
 
# i variable is to give unique name to images
i = 1
 
wait = 0
 
# Open the camera
video = cv2.VideoCapture(0)
 
 
while True:
    # Read video by read() function and it
    # will extract and  return the frame
    ret, img = video.read()
 
    # Put current DateTime on each frame
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(img, str(datetime.now()), (20, 40),
                font, 2, (255, 255, 255), 2, cv2.LINE_AA)
 
    # Display the image
    cv2.imshow('live video', img)
 
    # wait for user to press any key
    key = cv2.waitKey(100)
 
    # wait variable is to calculate waiting time
    wait = wait+100
 
    if key == ord('q'):
        break
    # when it reaches to 5000 milliseconds
    # we will save that frame in given folder
    if wait == 5000:
        filename = 'Frame_'+str(i)+'.jpg'
         
        # Save the images in given path
        cv2.imwrite(filename, img)
        i = i+1
        wait = 0
 
# close the camera
video.release()
 
# close open windows
cv2.destroyAllWindows()