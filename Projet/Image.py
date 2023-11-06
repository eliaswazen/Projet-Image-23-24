import cv2
import numpy as np


class Image:
    def __init__(self):
        self.largeur = 0
        self.hauteur = 0
        self.contenu = None

    def __init__(self, chemin_image):
        self.contenu = cv2.imread(chemin_image)
        dimensions = self.contenu.shape
        self.hauteur = dimensions[0]
        self.largeur = dimensions[1]

    def charger_image(self, chemin_image):
        self.contenu = cv2.imread(chemin_image)
        dimensions = self.contenu.shape
        self.hauteur = dimensions[0]
        self.largeur = dimensions[1]

    def afficher_image(self):
        if self.contenu is not None:
            cv2.imshow('Image', self.contenu)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Aucune image chargée.")

    def redimention(self, goalLargeur, goalHauteur):
        image_tampon = cv2.resize(self.contenu, (goalLargeur, goalHauteur))
        self.contenu = image_tampon

    def rotation(self):
        image_list = []

        image_rotate_90 = cv2.rotate(self.contenu, cv2.ROTATE_90_CLOCKWISE)
        image_rotate_180 = cv2.rotate(self.contenu, cv2.ROTATE_180)
        image_rotate_270 = cv2.rotate(self.contenu, cv2.ROTATE_90_COUNTERCLOCKWISE)

        image_list.append(image_rotate_90)
        image_list.append(image_rotate_180)
        image_list.append(image_rotate_270)

        output_directory = 'Images/'

        for i, image in enumerate(image_list):
            filename = f'{output_directory}image_{i}.jpeg'  # You can change the file format if needed
            cv2.imwrite(filename, image)

        print('Images saved successfully.')


    def objectDetection(self):

        video_src = 'pedestrians.avi'
        cap = cv2.VideoCapture(video_src)

        bike_cascade = cv2.CascadeClassifier('pedestrian.xml')

        while True:
            ret, img = cap.read()

            if (type(img) == type(None)):
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bike = bike_cascade.detectMultiScale(gray, 1.3, 2)

            for (a, b, c, d) in bike:
                cv2.rectangle(img, (a, b), (a + c, b + d), (0, 255, 210), 4)

            cv2.imshow('video', img)

            if cv2.waitKey(33) == 27:
                break

        cv2.destroyAllWindows()



        # object_cascade = cv2.CascadeClassifier('pedestrian.xml')
        # gray = cv2.cvtColor(self.contenu, cv2.COLOR_BGR2GRAY)
        # objects = object_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
        #
        # for (x, y, w, h) in objects:
        #     cv2.rectangle(self.contenu, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #
        # cv2.imshow('Object Detection', self.contenu)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # gray_image = cv2.cvtColor(self.contenu, cv2.COLOR_BGR2GRAY)
        #
        # blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        # cv2.imshow('Object Detection', self.contenu)
        #
        # _, binary_image = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY)
        #
        # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        # # Draw rectangles around the detected objects
        # for contour in contours:
        #     if cv2.contourArea(contour) > 1000:
        #         x, y, w, h = cv2.boundingRect(contour)
        #         cv2.rectangle(self.contenu, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #
        # # Display the result
        # cv2.imshow('Object Detection', self.contenu)
        # cv2.waitKey(0)

