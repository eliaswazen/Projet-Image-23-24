import cv2

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
