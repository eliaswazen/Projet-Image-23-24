import cv2
import numpy as np


class CarPedestrian:
    def __init__(self):
        self.contenu = None

    def __init__(self, chemin_video):
        self.contenu = cv2.VideoCapture(chemin_video)



print('Project Topic : Vehicle Classification')
print('Research Internship on Machine learning using Images')
print('By Aditya Yogish Pai and Aditya Baliga B')

cascade_src = 'XML/cars.xml'
cascade_pieton_src = 'XML/pedestrian.xml'

video_src = 'Videos/video2.mov'

cap = cv2.VideoCapture(video_src)

car_cascade = cv2.CascadeClassifier(cascade_src)
pieton_cascade = cv2.CascadeClassifier(cascade_pieton_src)

while True:
    ret, img = cap.read()

    if (type(img) == type(None)):
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.1, 2)
    pedestrians = pieton_cascade.detectMultiScale(gray, 1.1, 2)

    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
        label = f"Voiture"  # Create the label
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        label = f"Pieton"  # Create the label
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 210), 2)

    cv2.imshow('video', img)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()