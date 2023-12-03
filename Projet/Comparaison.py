# import cv2
#
# # Charger le fichier cascade pour la détection des visages
# cascade_path = 'XML/cars.xml'
# voitures_cascade = cv2.CascadeClassifier(cascade_path)
#
# for i in range(16):
#     # Charger l'image sur laquelle appliquer la détection
#     image_path = 'ImagesComp/' +  str(i + 1) + '.jpg'
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Appliquer la détection des voitures
#     detect = voitures_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=0, minSize=(30, 30))
#
#     # Dessiner des rectangles autour des voitures détectés
#     plus_grand_visage = None
#     plus_grande_taille = 0
#
#     for (x, y, w, h) in detect:
#         taille_actuelle = w * h
#         if taille_actuelle > plus_grande_taille:
#             plus_grande_taille = taille_actuelle
#             plus_grand_visage = (x, y, w, h)
#
#     # Dessiner un rectangle autour du plus grand visage détecté
#     if plus_grand_visage is not None:
#         (x, y, w, h) = plus_grand_visage
#         cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
#
#     print(x, y, x+w, y+h)

# cv2.imshow('Face Detection', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np

for i in range(16):
    # Charger l'image sur laquelle appliquer la détection
    image_path = 'ImagesComp/' +  str(i + 1) + '.jpg'
    image = cv2.imread(image_path)

    net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")

    with open("COCO/coco.names", "r")  as f:
        classes = f.read().strip().split('\n')

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()


    detections = net.forward(output_layers_names)

    max_confidence = 0
    bounding_box = None


    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > max_confidence:
                max_confidence = confidence

                # Calculer les coordonnées du rectangle
                center_x, center_y, width, height = map(int, obj[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]]))
                x, y = center_x - width // 2, center_y - height // 2

                # Enregistrer les coordonnées du rectangle avec la plus haute confiance
                bounding_box = (x, y, x + width, y + height)

    print(bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3])
    # if bounding_box is not None:
    #     color = (0, 255, 0)  # Couleur en BGR
    #     cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), color, 2)
    #     cv2.putText(image, f'Object: {classes[class_id]} - Confidence: {max_confidence:.2f}', (bounding_box[0], bounding_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# cv2.imshow('YOLO Object Detection', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()