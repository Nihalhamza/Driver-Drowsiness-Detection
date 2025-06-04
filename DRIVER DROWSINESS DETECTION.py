import cv2
import os
import tensorflow as tf
import numpy as np
from pygame import mixer
import time

mixer.init()
sound = mixer.Sound(r"C:\Users\nihal\OneDrive\Desktop\DS PROJECTS\DL GROUP PROJECT\alarm.wav")

face = cv2.CascadeClassifier(r"C:\Users\nihal\OneDrive\Desktop\DS PROJECTS\DL GROUP PROJECT\haarcascade_frontalface_alt.xml")
leye = cv2.CascadeClassifier(r"C:\Users\nihal\OneDrive\Desktop\DS PROJECTS\DL GROUP PROJECT\haarcascade_lefteye_2splits.xml")
reye = cv2.CascadeClassifier(r"C:\Users\nihal\OneDrive\Desktop\DS PROJECTS\DL GROUP PROJECT\haarcascade_righteye_2splits.xml")

model = tf.keras.models.load_model(r"C:\Users\nihal\OneDrive\Desktop\DS PROJECTS\DL GROUP PROJECT\drowsiness_detection_cnn.h5")

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
alarm_on = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    rpred_class = 1
    lpred_class = 1

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2RGB)
        r_eye = cv2.resize(r_eye, (64, 64))
        r_eye = r_eye / 255.0
        r_eye = np.expand_dims(r_eye, axis=-1)
        r_eye = np.expand_dims(r_eye, axis=0)

        rpred = model.predict(r_eye)
        rpred_class = np.argmax(rpred, axis=1)[0]
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2RGB)
        l_eye = cv2.resize(l_eye, (64, 64))
        l_eye = l_eye / 255.0
        l_eye = np.expand_dims(l_eye, axis=-1)
        l_eye = np.expand_dims(l_eye, axis=0)

        lpred = model.predict(l_eye)
        lpred_class = np.argmax(lpred, axis=1)[0]
        break

    if rpred_class == 0 and lpred_class == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score = max(0, score - 1)
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 15:
        if not alarm_on:
            sound.play(-1)
            alarm_on = True
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 2)
    else:
        if alarm_on:
            sound.stop()
            alarm_on = False


    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()