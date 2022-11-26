import os
import cv2
import numpy as np
from utils import cv2_img_add_text,index2emotion
from blazeface import blaze_detect



def face_detect(img_path, model_selection="default"):
    #Detect faces in the image
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if model_selection == "default":
        face_cascade = cv2.CascadeClassifier('./dataset/params/haarcascade_frontalface_alt.xml')
        faces = face_cascade.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(30, 30)
        )
    elif model_selection == "blazeface":
        faces = blaze_detect(img)
    else:
        raise NotImplementedError("this face detector is not supported!")

    return img, img_gray, faces


def generate_faces(face_img, img_size=48):
    #Augment the detected face
    face_img = face_img / 255.
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = list()
    resized_images.append(face_img[:, :])
    resized_images.append(face_img[2:45, :])
    resized_images.append(cv2.flip(face_img[:, :], 1))
    resized_images.append(face_img[0:45, 0:45])
    resized_images.append(face_img[2:47, 0:45])
    resized_images.append(face_img[2:47, 2:47])

    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    resized_images = np.array(resized_images)
    return resized_images


def predict_expression(img_path, model):
    #Predict n faces in the image
    border_color = (0, 255, 0)
    font_color = (255, 255, 255)
    img, img_gray, faces = face_detect(img_path, 'blazeface')
    if len(faces) == 0:
        return 'no', [0, 0, 0, 0, 0, 0, 0, 0]
    #enumerate every face
    emotions = []
    result_possibilities = []
    for (x, y, w, h) in faces:
        face_img_gray = img_gray[y:y + h + 10, x:x + w + 10]
        faces_img_gray = generate_faces(face_img_gray)
        #average the results of predictions
        results = model.predict(faces_img_gray)
        result_sum = np.sum(results, axis=0).reshape(-1)
        label_index = np.argmax(result_sum, axis=0)
        emotion = index2emotion(label_index)
        #draw the rectangle for the interests region    
        cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), border_color, thickness=2)
        img = cv2_img_add_text(img, emotion, x + 30, y + 30, font_color)
        emotions.append(emotion)
        result_possibilities.append(result_sum)
    #Output the result image to the "results" folder
    if not os.path.exists("./results"):
        os.makedirs("./results")
    cv2.imwrite('./results/rst.png', img)
    return emotions[0], result_possibilities[0]