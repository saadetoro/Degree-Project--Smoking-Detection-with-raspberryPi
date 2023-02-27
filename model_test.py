import cv2
import tensorflow as tf


def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

CATEGORIES = ["Smoking", "Not-Smoking"]

model = tf.keras.models.load_model("CNN.model")
image = prepare('gg0081.jpg') #your image path"test.jpg"
prediction = model.predict([image])
prediction = list(prediction[0])
print(CATEGORIES[prediction.index(max(prediction))])