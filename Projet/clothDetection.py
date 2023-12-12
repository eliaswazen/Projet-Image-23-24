# import cv2
# import numpy as np
#
# # Load YOLO
# net = cv2.dnn.readNet("yolo/yolov3-tiny.weights", "yolo/yolov3-tiny.cfg")
# classes = []
# with open("COCO/coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]
# layer_names = net.getUnconnectedOutLayersNames()
#
# # Load accessory image
# hat_img = cv2.imread("Images/hat.png", -1)
#
# # Load video
# cap = cv2.VideoCapture("Videos/video2.mov")
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     height, width, _ = frame.shape
#
#     # YOLO preprocessing
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(layer_names)
#
#     # Detection and drawing boxes
#     class_ids = []
#     confidences = []
#     boxes = []
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5 and class_id == 0:  # Class 0 corresponds to person in COCO dataset
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
#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#
#             # Add hat accessory (you can replace this with your own accessory)
#             hat_width = int(1.5 * w)
#             hat_height = int(0.7 * h)
#             hat_x = x - int((hat_width - w) / 2)
#             hat_y = y - hat_height
#
#             # Resize the hat image to fit the person's head
#             hat_resized = cv2.resize(hat_img, (hat_width, hat_height))
#
#             # Overlay the hat onto the frame
#             for c in range(3):
#                 frame[hat_y:hat_y + hat_height, hat_x:hat_x + hat_width, c] = (
#                         frame[hat_y:hat_y + hat_height, hat_x:hat_x + hat_width, c] * (
#                             1 - hat_resized[:, :, 3] / 255.0) +
#                         hat_resized[:, :, c] * (hat_resized[:, :, 3] / 255.0)
#                 )
#
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # Display the resulting frame
#     cv2.imshow('Video', frame)
#
#     # Break the loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()

# STARTTTTTTT

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load YOLO
net = cv2.dnn.readNet("yolo/yolov3-tiny.weights", "yolo/yolov3-tiny.cfg")
classes = []
with open("COCO/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getUnconnectedOutLayersNames()

# Initialize accessory variables
show_hat = True
hat_img = cv2.imread("Images/hat.png", -1)

# Load video
cap = cv2.VideoCapture("Videos/pedestrians.avi")
# cap = cv2.VideoCapture(0)

# Create a tkinter window
root = tk.Tk()
root.title("Accessory App")


# Function to toggle showing the hat
def toggle_hat():
    global show_hat
    show_hat = not show_hat


# Function to upload a new hat image
def upload_hat():
    global hat_img
    file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
    hat_img = cv2.imread(file_path, -1)


# Button to toggle hat visibility
toggle_button = tk.Button(root, text="Toggle Hat", command=toggle_hat)
toggle_button.pack()

# Button to upload a new hat image
upload_button = tk.Button(root, text="Upload Hat Image", command=upload_hat)
upload_button.pack()


# Function to update the GUI window
def update_gui():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        root.destroy()
        return

    height, width, _ = frame.shape

    # YOLO preprocessing
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Detection and drawing boxes
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class 0 corresponds to person in COCO dataset
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

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]

            if show_hat:
                # Add hat accessory (you can replace this with your own accessory)
                hat_width = int(1.5 * w)
                hat_height = int(0.7 * h)
                hat_x = x - int((hat_width - w) / 2)
                hat_y = y - hat_height

                # Resize the hat image to fit the person's head
                hat_resized = cv2.resize(hat_img, (hat_width, hat_height))

                # Overlay the hat onto the frame
                for c in range(3):
                    frame[hat_y:hat_y + hat_height, hat_x:hat_x + hat_width, c] = (
                            frame[hat_y:hat_y + hat_height, hat_x:hat_x + hat_width, c] * (
                                1 - hat_resized[:, :, 3] / 255.0) +
                            hat_resized[:, :, c] * (hat_resized[:, :, 3] / 255.0)
                    )

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    panel.imgtk = imgtk
    panel.config(image=imgtk)

    root.after(10, update_gui)


# Create a tkinter label for displaying video frames
panel = tk.Label(root)
panel.pack()

# Start the GUI window update loop
update_gui()

# Run the tkinter main loop
root.mainloop()

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

#ENDDDDD

# import cv2
# import numpy as np
# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk
#
# # Load YOLO
# net = cv2.dnn.readNet("yolo/yolov3-tiny.weights", "yolo/yolov3-tiny.cfg")
# classes = []
# with open("COCO/coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]
# layer_names = net.getUnconnectedOutLayersNames()
#
# # Initialize accessory variables
# show_hat = True
# show_handbag = True
# show_clothes = True
# show_shoes = True
#
# hat_img = cv2.imread("Images/hat.png", -1)
# handbag_img = cv2.imread("Images/handbag.png", -1)
# clothes_img = cv2.imread("Images/clothes.png", -1)
# shoes_img = cv2.imread("Images/shoes.png", -1)
#
# # Load video
# # cap = cv2.VideoCapture("Videos/pedestrians.avi")
#
# # Open video from device (webcam)
# cap = cv2.VideoCapture(0)
#
# # Create a tkinter window
# root = tk.Tk()
# root.title("Accessory App")
#
#
# # Function to toggle showing accessories
# def toggle_accessories(accessory_type):
#     global show_hat, show_handbag, show_clothes, show_shoes
#     if accessory_type == "hat":
#         show_hat = not show_hat
#     elif accessory_type == "handbag":
#         show_handbag = not show_handbag
#     elif accessory_type == "clothes":
#         show_clothes = not show_clothes
#     elif accessory_type == "shoes":
#         show_shoes = not show_shoes
#
#
# # Function to upload a new accessory image
# def upload_accessory(accessory_type):
#     global hat_img, handbag_img, clothes_img, shoes_img
#     file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
#     accessory_img = cv2.imread(file_path, -1)
#
#     if accessory_type == "hat":
#         hat_img = accessory_img
#     elif accessory_type == "handbag":
#         handbag_img = accessory_img
#     elif accessory_type == "clothes":
#         clothes_img = accessory_img
#     elif accessory_type == "shoes":
#         shoes_img = accessory_img
#
#
# # Button to toggle hat visibility
# toggle_hat_button = tk.Button(root, text="Toggle Hat", command=lambda: toggle_accessories("hat"))
# toggle_hat_button.pack()
#
# # Button to toggle handbag visibility
# toggle_handbag_button = tk.Button(root, text="Toggle Handbag", command=lambda: toggle_accessories("handbag"))
# toggle_handbag_button.pack()
#
# # Button to toggle clothes visibility
# toggle_clothes_button = tk.Button(root, text="Toggle Clothes", command=lambda: toggle_accessories("clothes"))
# toggle_clothes_button.pack()
#
# # Button to toggle shoes visibility
# toggle_shoes_button = tk.Button(root, text="Toggle Shoes", command=lambda: toggle_accessories("shoes"))
# toggle_shoes_button.pack()
#
# # Button to upload a new accessory image for hat
# upload_hat_button = tk.Button(root, text="Upload Hat Image", command=lambda: upload_accessory("hat"))
# upload_hat_button.pack()
#
# # Button to upload a new accessory image for handbag
# upload_handbag_button = tk.Button(root, text="Upload Handbag Image", command=lambda: upload_accessory("handbag"))
# upload_handbag_button.pack()
#
# # Button to upload a new accessory image for clothes
# upload_clothes_button = tk.Button(root, text="Upload Clothes Image", command=lambda: upload_accessory("clothes"))
# upload_clothes_button.pack()
#
# # Button to upload a new accessory image for shoes
# upload_shoes_button = tk.Button(root, text="Upload Shoes Image", command=lambda: upload_accessory("shoes"))
# upload_shoes_button.pack()
#
#
# # Function to update the GUI window
# def update_gui():
#     # Read a frame from the video
#     ret, frame = cap.read()
#     if not ret:
#         root.destroy()
#         return
#
#     height, width, _ = frame.shape
#
#     # YOLO preprocessing
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(layer_names)
#
#     # Detection and drawing boxes
#     class_ids = []
#     confidences = []
#     boxes = []
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5 and class_id == 0:  # Class 0 corresponds to person in COCO dataset
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
#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#
#             # Overlay accessories
#             if show_hat:
#                 overlay_accessory(frame, hat_img, x, y, w, h, scale_width=1.5, scale_height=0.7, offset_y=-int(0.3 * h))
#
#             if show_handbag:
#                 overlay_accessory(frame, handbag_img, x, y, w, h, scale_width=1.2, scale_height=0.7,
#                                   offset_y=int(0.3 * h))
#
#             if show_clothes:
#                 overlay_accessory(frame, clothes_img, x, y, w, h, scale_width=1.0, scale_height=1.0,
#                                   offset_y=int(0.5 * h))
#
#             if show_shoes:
#                 overlay_accessory(frame, shoes_img, x, y, w, h, scale_width=1.2, scale_height=0.5,
#                                   offset_y=int(0.6 * h))
#
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # Display the resulting frame
#     cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(cv2image)
#     imgtk = ImageTk.PhotoImage(image=img)
#     panel.imgtk = imgtk
#     panel.config(image=imgtk)
#
#     root.after(10, update_gui)
#
#
# # Function to overlay an accessory onto the frame
# # Function to overlay an accessory onto the frame
# def overlay_accessory(frame, accessory_img, x, y, w, h, scale_width, scale_height, offset_y):
#     accessory_width = int(scale_width * w)
#     accessory_height = int(scale_height * h)
#     accessory_x = x - int((accessory_width - w) / 2)
#     accessory_y = y + offset_y
#
#     # Resize the accessory image to fit the accessory dimensions
#     accessory_resized = cv2.resize(accessory_img, (accessory_width, accessory_height))
#
#     # Extract the alpha channel (transparency) from the accessory image
#     alpha_channel = accessory_resized[:, :, 3] / 255.0
#
#     # Calculate the region to overlay the accessory on the frame
#     overlay_region = frame[accessory_y:accessory_y + accessory_height, accessory_x:accessory_x + accessory_width]
#
#     # Ensure that overlay_region dimensions match accessory_resized dimensions
#     overlay_region = cv2.resize(overlay_region, (accessory_width, accessory_height))
#
#     # Perform the overlay using the alpha channel for transparency
#     for c in range(3):
#         overlay_region[:, :, c] = (
#             overlay_region[:, :, c] * (1 - alpha_channel) +
#             accessory_resized[:, :, c] * alpha_channel
#         )
#
#
# # Create a tkinter label for displaying video frames
# panel = tk.Label(root)
# panel.pack()
#
# # Start the GUI window update loop
# update_gui()
#
# # Run the tkinter main loop
# root.mainloop()
#
# # Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()
