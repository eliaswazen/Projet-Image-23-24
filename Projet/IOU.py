def lire_bbox_de_fichier(chemin_fichier):
    with open(chemin_fichier, 'r') as file:
        lignes = file.readlines()
        bounding_boxes = [tuple(map(int, ligne.strip().split())) for ligne in lignes]
    return bounding_boxes

def calculer_iou(bbox_reference, bbox_detection):
    # Bounding box de référence : (x1, y1, x2, y2)
    x1_ref, y1_ref, x2_ref, y2_ref = bbox_reference

    # Bounding box de détection : (x1, y1, x2, y2)
    x1_det, y1_det, x2_det, y2_det = bbox_detection

    # Calcul de l'aire de l'intersection
    intersection_area = max(0, min(x2_ref, x2_det) - max(x1_ref, x1_det)) * max(0, min(y2_ref, y2_det) - max(y1_ref,
                                                                                                             y1_det))

    # Calcul de l'aire de l'union
    aire_reference = (x2_ref - x1_ref) * (y2_ref - y1_ref)
    aire_detection = (x2_det - x1_det) * (y2_det - y1_det)
    union_area = aire_reference + aire_detection - intersection_area

    # Calcul de l'IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou

# Chemins vers les fichiers de bounding boxes de référence et de détection
chemin_fichier_reference = 'ImagesComp/reference.txt'
chemin_fichier_detection = 'ImagesComp/Haar.txt'

# Lire les bounding boxes à partir des fichiers
bounding_boxes_reference = lire_bbox_de_fichier(chemin_fichier_reference)
bounding_boxes_detection = lire_bbox_de_fichier(chemin_fichier_detection)

# Calculer l'IoU moyen
iou_total = 0
nombre_bbox = len(bounding_boxes_reference)

for bbox_ref, bbox_det in zip(bounding_boxes_reference, bounding_boxes_detection):
    iou_total += calculer_iou(bbox_ref, bbox_det)

iou_moyen = iou_total / nombre_bbox if nombre_bbox > 0 else 0

print(f"IoU Moyen : {iou_moyen}")