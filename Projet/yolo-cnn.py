# import cv2
# import numpy as np
#
# # Load YOLO
# net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
#
# layers = net.getLayerNames()
# print("Layers:", layers)
# # Debugging: Print unconnected output layers
# unconnected_out_layers = net.getUnconnectedOutLayers()
# print("Unconnected Output Layers:", unconnected_out_layers)
#
#
# output_layers = [int(i) for i in unconnected_out_layers]
#
# # Initialize variables
# font = cv2.FONT_HERSHEY_PLAIN
# tracking_objects = []
#
# # Mouse callback function for ROI selection
# def select_roi(event, x, y, flags, param):
#     global tracking_objects
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # Add selected ROI to the list of tracking objects
#         tracking_objects.append((x, y, 100, 100))  # Example: 100x100 ROI
#         print("Selected ROI:", (x, y, 100, 100))
#
# # Open video capture
# cap = cv2.VideoCapture('Videos/video2.mov')  # Use 0 for webcam
#
# # Create window and set mouse callback
# cv2.namedWindow('Object Detection')
# cv2.setMouseCallback('Object Detection', select_roi)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     height, width, channels = frame.shape
#
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)
#
#     # Process the detection results
#     class_ids = []
#     confidences = []
#     boxes = []
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#
#     # Draw bounding boxes and labels
#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#             label = str(class_ids[i])
#             confidence = confidences[i]
#             color = (0, 255, 0)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), font, 1, color, 1)
#
#     # Draw tracking rectangles based on user-selected ROIs
#     for roi in tracking_objects:
#         x, y, w, h = roi
#         color = (255, 0, 0)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#
#     # Display the frame with detected objects
#     cv2.imshow('Object Detection', frame)
#
#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release video capture and close windows
# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")

layers = net.getLayerNames()
print("Layers:", layers)
# Debugging: Print unconnected output layers
unconnected_out_layers = net.getUnconnectedOutLayers()
print("Unconnected Output Layers:", unconnected_out_layers)

output_layers = [int(i) for i in unconnected_out_layers]

# Define class names
class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
               "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
               "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
               "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
               "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
               "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
               "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
               "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Initialize variables
font = cv2.FONT_HERSHEY_PLAIN
tracking_objects = []

# Mouse callback function for ROI selection
def select_roi(event, x, y, flags, param):
    global tracking_objects
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add selected ROI to the list of tracking objects
        tracking_objects.append((x, y, 100, 100))  # Example: 100x100 ROI
        print("Selected ROI:", (x, y, 100, 100))

# Open video capture
cap = cv2.VideoCapture('Videos/video2.mov')  # Use 0 for webcam

# Create window and set mouse callback
cv2.namedWindow('Object Detection')
cv2.setMouseCallback('Object Detection', select_roi)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process the detection results
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = class_names[class_ids[i]]
            confidence = confidences[i]
            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), font, 1, color, 1)

    # Draw tracking rectangles based on user-selected ROIs
    for roi in tracking_objects:
        x, y, w, h = roi
        color = (255, 0, 0)  # Blue color for tracking rectangles
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display the frame with detected objects
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
