import cv2
import numpy as np


class Pieton:
    def __init__(self):
        self.contenu = None

    def __init__(self, chemin_video):
        self.contenu = cv2.VideoCapture(chemin_video)



print('Project Topic : Vehicle Classification')
print('Research Internship on Machine learning using Images')
print('By Aditya Yogish Pai and Aditya Baliga B')

video_src = 'Videos/pedestrians.avi'

cap = cv2.VideoCapture(video_src)

bike_cascade = cv2.CascadeClassifier('XML/pedestrian.xml')

while True:
    ret, img = cap.read()

    if (type(img) == type(None)):
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bikes = bike_cascade.detectMultiScale(gray, 1.3, 2)

    for (a, b, c, d) in bikes:
        cv2.rectangle(img, (a, b), (a + c, b + d), (0, 255, 210), 4)
        # Calculate accuracy or matching here (you need to define this logic)
        accuracy = 0.75  # Replace this with your accuracy calculation

        label = f"Pieton"  # Create the label
        cv2.putText(img, label, (a, b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 210), 2)

    cv2.imshow('video', img)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
