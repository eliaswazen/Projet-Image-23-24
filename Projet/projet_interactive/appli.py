# import cv2
# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk
# import numpy as np
#
#
# class VideoApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Object Detection and Tracking")
#
#         self.video_source = None
#         self.vid = None
#         self.current_detection_method = None
#         self.net = None
#         self.cascade_src = None
#         self.cascade_pieton_src = None
#         self.model = None
#
#         self.frame_width = 640
#         self.frame_height = 480
#
#         self.margin = 10  # Adjust the margin as needed
#
#         # Create frame for 1st buttons
#         self.button_hy_frame = tk.Frame(root)
#         self.button_hy_frame.pack(pady=10)
#
#
#
#         self.btn_haar = tk.Button(self.button_hy_frame, text="Haar", command=self.haar, bg="lightgray")
#         self.btn_yolo = tk.Button(self.button_hy_frame, text="Yolo", command=self.yolo, bg="lightgray")
#         self.btn_stop = tk.Button(self.button_hy_frame, text="Stop", command=self.stop_detection, bg="lightgray")
#
#         self.btn_haar.grid(row=0, column=0, padx=self.margin)
#         self.btn_yolo.grid(row=0, column=1, padx=self.margin)
#         self.btn_stop.grid(row=0, column=2, padx=self.margin)
#
#         # Create frame for buttons
#         self.button_frame = tk.Frame(root)
#         self.button_frame.pack(pady=0)
#
#         self.btn_load_video = tk.Button(self.button_frame, text="Load Video", command=self.load_video)
#         self.btn_open_webcam = tk.Button(self.button_frame, text="Open Webcam", command=self.open_webcam)
#         self.btn_switch = tk.Button(self.button_frame, text="Switch", command=self.switch_video_source)
#
#         self.label_detection_method = tk.Label(self.button_frame, text="Detection Method: None", padx=10)
#         self.label_detection_method.grid(row=0, column=4)
#
#         self.btn_load_video.grid(row=0, column=0, padx=self.margin)
#         self.btn_open_webcam.grid(row=0, column=1, padx=self.margin)
#         self.btn_switch.grid(row=0, column=2, padx=self.margin)
#
#         # Create frame for canvas
#
#         self.canvas_frame = tk.Frame(root, bg="gray")
#         self.canvas_frame.pack()
#
#         self.canvas = tk.Canvas(self.canvas_frame, width=self.frame_width, height=self.frame_height, bg="gray")
#         self.canvas.pack()
#
#         self.is_webcam = False
#
#         self.update()
#
#     def update_detection_label(self):
#         detection_method = "None"
#         if self.current_detection_method == "Haar":
#             detection_method = "Haar"
#         elif self.current_detection_method == "Yolo":
#             detection_method = "Yolo"
#         elif self.current_detection_method == "Stop":
#             detection_method = "Stop"
#         self.label_detection_method.config(text=f"Detection Method: {detection_method}")
#
#     def stop_detection(self):
#         print("Detection Stopped")
#         self.current_detection_method = "Stop"
#         self.update_detection_label()
#         self.btn_haar.config(bg="lightgray")  # Reset Haar button color
#         self.btn_yolo.config(bg="lightgray")  # Reset YOLO button color
#         self.btn_stop.config(bg="lightblue")
#
#     def haar(self):
#         print("Hello, this is haar!")
#         self.current_detection_method = "Haar"
#         self.update_detection_label()
#         self.btn_haar.config(bg="lightblue")  # Change button color
#         self.btn_yolo.config(bg="lightgray")  # Reset YOLO button color
#         self.btn_stop.config(bg="lightgray")
#
#         self.cascade_src = '../XML/cars.xml'
#         self.cascade_pieton_src = '../XML/pedestrian.xml'
#
#     def haar_video(self, frame):
#         car_cascade = cv2.CascadeClassifier(self.cascade_src)
#         pieton_cascade = cv2.CascadeClassifier(self.cascade_pieton_src)
#
#         cars = car_cascade.detectMultiScale(frame, 1.1, 2)
#         pedestrians = pieton_cascade.detectMultiScale(frame, 1.2, 1)
#
#         for (x, y, w, h) in cars:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             label = f"Voiture"  # Create the label
#             cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#         for (x, y, w, h) in pedestrians:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
#             label = f"Pieton"  # Create the label
#             cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 210), 2)
#
#     def yolo(self):
#         print("Hello, this is yolo!")
#         self.current_detection_method = "Yolo"
#         self.update_detection_label()
#         self.btn_haar.config(bg="lightgray")  # Reset Haar button color
#         self.btn_yolo.config(bg="lightblue")  # Change button color
#         self.btn_stop.config(bg="lightgray")
#
#         # Load YOLO
#         self.net = cv2.dnn.readNet("../yolo/yolov3.weights", "../yolo/yolov3.cfg")
#         self.classes = []
#
#         if self.is_webcam:
#             self.model = cv2.dnn_DetectionModel(self.net)
#             self.model.setInputParams(size=(320, 320), scale=1 / 255)
#             # Load class lists
#             with open("../COCO/coco.names", "r") as file_object:
#                 for class_name in file_object.readlines():
#                     class_name = class_name.strip()
#                     self.classes.append(class_name)
#         else:
#             # Load classes
#             with open("../COCO/coco.names", "r") as f:
#                 self.classes = [line.strip() for line in f]
#
#
#
#     def yolo_video(self,frame):
#         layer_names = self.net.getLayerNames()
#         output_layers = []
#
#         unconnected_layers = self.net.getUnconnectedOutLayers()
#
#         if unconnected_layers is not None and len(unconnected_layers) > 0:
#             for i in unconnected_layers:
#                 if isinstance(i, (list, tuple)) and i and isinstance(i[0], int) and 0 <= i[0] - 1 < len(
#                         layer_names):
#                     output_layers.append(layer_names[i[0] - 1])
#
#         height, width, channels = frame.shape
#
#         # Detecting objects
#         blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#         self.net.setInput(blob)
#         outs = self.net.forward(output_layers)
#         # Information to display on the image
#         class_ids = []
#         confidences = []
#         boxes = []
#
#         # Process each detection
#         for out in outs:
#             for detection in out:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]
#                 if confidence > 0.5:  # Confidence threshold
#                     # Object detected
#                     center_x = int(detection[0] * width)
#                     center_y = int(detection[1] * height)
#                     w = int(detection[2] * width)
#                     h = int(detection[3] * height)
#
#                     # Rectangle coordinates
#                     x = int(center_x - w / 2)
#                     y = int(center_y - h / 2)
#
#                     boxes.append([x, y, w, h])
#                     confidences.append(float(confidence))
#                     class_ids.append(class_id)
#
#         # Apply non-max suppression to remove overlapping bounding boxes
#         indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#
#         # Draw bounding boxes and labels on the image
#         for i in range(len(boxes)):
#             if i in indexes:
#                 x, y, w, h = boxes[i]
#                 label = str(self.classes[class_ids[i]])
#                 confidence = confidences[i]
#                 color = (0, 255, 0)  # Green
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#                 cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                             color, 2)
#
#     def yolo_webcam(self,frame):
#
#         (class_ids, scores, bboxes) = self.model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
#         for class_id, score, bbox in zip(class_ids, scores, bboxes):
#             (x, y, w, h) = bbox
#             class_name = self.classes[class_id]
#             random_c = np.random.randint(256, size=3)
#             color = (int(random_c[0]), int(random_c[1]), int(random_c[2]))
#             cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
#
#     def load_video(self):
#         file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
#         if file_path:
#             self.video_source = file_path
#             self.vid = cv2.VideoCapture(self.video_source)
#             self.is_webcam = False
#             self.canvas.config(bg="gray")
#
#     def open_webcam(self):
#         self.video_source = 0
#         self.vid = cv2.VideoCapture(self.video_source)
#         self.is_webcam = True
#         self.canvas.config(bg="gray")
#
#     def switch_video_source(self):
#         if self.vid:
#             self.vid.release()
#         if self.is_webcam:
#             self.load_video()
#         else:
#             self.open_webcam()
#
#
#
#     def update(self):
#         if self.vid is not None:
#             ret, frame = self.vid.read()
#             if ret:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame = cv2.resize(frame, (self.frame_width, self.frame_height))
#                 if self.vid and self.current_detection_method == "Haar" and self.is_webcam ==False:
#                     print("haar vid")
#                     self.haar_video(frame)
#
#                 elif self.vid and self.current_detection_method == "Yolo" and self.is_webcam ==False:
#                     print("Yolo vid")
#                     self.yolo_video(frame)
#
#                 if self.is_webcam and self.current_detection_method == "Haar":
#                     print("haar is_webcam")
#
#                 elif self.is_webcam and self.current_detection_method == "Yolo":
#                     print("Yolo is_webcam")
#                     self.yolo_webcam(frame)
#
#                 elif self.current_detection_method == "Stop":
#
#                     pass
#
#                 self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
#                 self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
#             else:
#                 self.canvas.config(bg="gray")
#         self.root.after(100, self.update)
#
#     def __del__(self):
#         if self.vid is not None and self.vid.isOpened():
#             self.vid.release()
#
# root = tk.Tk()
# app = VideoApp(root)
# root.mainloop()


import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
from datetime import datetime
from PIL import Image, ImageGrab, ImageTk
import time



class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection and Tracking")

        self.video_source = None
        self.vid = None
        self.current_detection_method = None
        self.net = None
        self.cascade_src = None
        self.cascade_pieton_src = None
        self.model = None

        self.frame_width = 640
        self.frame_height = 480

        self.margin = 10  # Adjust the margin as needed

        # Create frame for 1st buttons
        self.button_hy_frame = tk.Frame(root)
        self.button_hy_frame.pack(pady=10)

        self.btn_haar = tk.Button(self.button_hy_frame, text="Haar", command=self.haar, bg="lightgray")
        self.btn_yolo = tk.Button(self.button_hy_frame, text="Yolo", command=self.yolo, bg="lightgray")
        self.btn_stop = tk.Button(self.button_hy_frame, text="Stop", command=self.stop_detection, bg="lightgray")

        self.btn_haar.grid(row=0, column=0, padx=self.margin)
        self.btn_yolo.grid(row=0, column=1, padx=self.margin)
        self.btn_stop.grid(row=0, column=2, padx=self.margin)

        # Create frame for buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=0)

        self.btn_load_video = tk.Button(self.button_frame, text="Load Video", command=self.load_video)
        self.btn_open_webcam = tk.Button(self.button_frame, text="Open Webcam", command=self.open_webcam)
        self.btn_switch = tk.Button(self.button_frame, text="Switch", command=self.switch_video_source)

        self.label_detection_method = tk.Label(self.button_frame, text="Detection Method: None", padx=10)
        self.label_detection_method.grid(row=0, column=4)

        self.btn_load_video.grid(row=0, column=0, padx=self.margin)
        self.btn_open_webcam.grid(row=0, column=1, padx=self.margin)
        self.btn_switch.grid(row=0, column=2, padx=self.margin)

        # Create frame for canvas and example buttons
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create frame for example buttons on the left side
        self.left_button_frame = tk.Frame(self.main_frame)
        self.left_button_frame.grid(row=0, column=0, padx=10, pady=10)

        # Webcam
        self.btn_example1 = tk.Button(self.left_button_frame, text="Example 1", command=self.example1, bg="lightgray")
        # Video deposer
        self.btn_pietons_detection = tk.Button(self.left_button_frame, text="Pietons", command=self.pietons_detection, bg="lightgray")

        # Hide example buttons at the start
        self.btn_example1.grid(row=0, pady=10, sticky=tk.W)
        self.btn_pietons_detection.grid(row=1, pady=10, sticky=tk.W)

        # Create frame for canvas on the right side
        self.canvas_frame = tk.Frame(self.main_frame)
        self.canvas_frame.grid(row=0, column=1, sticky=tk.NSEW)

        self.canvas = tk.Canvas(self.canvas_frame, width=self.frame_width, height=self.frame_height, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Create frame for control buttons under the canvas
        self.control_frame = tk.Frame(self.canvas_frame)
        self.control_frame.pack(pady=10)

        self.btn_pause = tk.Button(self.control_frame, text="Pause", command=self.toggle_play_pause)
        self.btn_screenshot = tk.Button(self.control_frame, text="Screenshot", command=self.take_screenshot)

        self.btn_pause.grid(row=0, column=0, padx=self.margin)
        self.btn_screenshot.grid(row=0, column=1, padx=self.margin)

        self.is_webcam = False
        self.is_paused = False  # Flag to track whether the video is paused
        self.is_pietons = False

        self.hide_example_buttons()  # Hide example buttons at the start

        self.update()

    def hide_example_buttons(self):
        self.btn_example1.grid_remove()
        self.btn_pietons_detection.grid_remove()
    def example1(self):
        # Add functionality for Example 1 button
        pass

    def pietons_detection(self):
        # Add functionality for Example 2 button
        self.is_pietons = not self.is_pietons
        print(self.is_pietons)

    def update_detection_label(self):
        detection_method = "None"
        if self.current_detection_method == "Haar":
            detection_method = "Haar"
        elif self.current_detection_method == "Yolo":
            detection_method = "Yolo"
        elif self.current_detection_method == "Stop":
            detection_method = "Stop"
        self.label_detection_method.config(text=f"Detection Method: {detection_method}")

    def stop_detection(self):
        print("Detection Stopped")
        self.current_detection_method = "Stop"
        self.update_detection_label()
        self.btn_haar.config(bg="lightgray")  # Reset Haar button color
        self.btn_yolo.config(bg="lightgray")  # Reset YOLO button color
        self.btn_stop.config(bg="lightblue")

    def haar(self):
        print("Hello, this is haar!")
        self.current_detection_method = "Haar"
        self.update_detection_label()
        self.btn_haar.config(bg="lightblue")  # Change button color
        self.btn_yolo.config(bg="lightgray")  # Reset YOLO button color
        self.btn_stop.config(bg="lightgray")

        self.cascade_src = '../XML/cars.xml'
        if self.is_pietons:
            self.cascade_pieton_src = '../XML/pedestrian.xml'

    def haar_video(self, frame):
        car_cascade = cv2.CascadeClassifier(self.cascade_src)
        if self.is_pietons:
            pieton_cascade = cv2.CascadeClassifier(self.cascade_pieton_src)

        cars = car_cascade.detectMultiScale(frame, 1.1, 2)
        if self.is_pietons:
            pedestrians = pieton_cascade.detectMultiScale(frame, 1.2, 1)

        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            label = f"Voiture"  # Create the label
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if self.is_pietons:
            for (x, y, w, h) in pedestrians:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                label = f"Pieton"  # Create the label
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 210), 2)

    def yolo(self):
        print("Hello, this is yolo!")
        self.current_detection_method = "Yolo"
        self.update_detection_label()
        self.btn_haar.config(bg="lightgray")  # Reset Haar button color
        self.btn_yolo.config(bg="lightblue")  # Change button color
        self.btn_stop.config(bg="lightgray")

        # Load YOLO
        self.net = cv2.dnn.readNet("../yolo/yolov3.weights", "../yolo/yolov3.cfg")
        self.classes = []

        if self.is_webcam:
            self.model = cv2.dnn_DetectionModel(self.net)
            self.model.setInputParams(size=(320, 320), scale=1 / 255)
            # Load class lists
            with open("../COCO/coco.names", "r") as file_object:
                for class_name in file_object.readlines():
                    class_name = class_name.strip()
                    self.classes.append(class_name)
        else:
            # Load classes
            with open("../COCO/coco.names", "r") as f:
                self.classes = [line.strip() for line in f]

    def yolo_video(self, frame):
        layer_names = self.net.getLayerNames()
        output_layers = []

        unconnected_layers = self.net.getUnconnectedOutLayers()

        if unconnected_layers is not None and len(unconnected_layers) > 0:
            for i in unconnected_layers:
                if isinstance(i, (list, tuple)) and i and isinstance(i[0], int) and 0 <= i[0] - 1 < len(
                        layer_names):
                    output_layers.append(layer_names[i[0] - 1])

        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(output_layers)
        # Information to display on the image
        class_ids = []
        confidences = []
        boxes = []

        # Process each detection
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Confidence threshold
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-max suppression to remove overlapping bounding boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes and labels on the image
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 2)

    def yolo_webcam(self, frame):
        (class_ids, scores, bboxes) = self.model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            (x, y, w, h) = bbox
            class_name = self.classes[class_id]
            random_c = np.random.randint(256, size=3)
            color = (int(random_c[0]), int(random_c[1]), int(random_c[2]))
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
        if file_path:
            self.video_source = file_path
            self.vid = cv2.VideoCapture(self.video_source)
            self.is_webcam = False
            self.canvas.config(bg="gray")
            self.btn_example1.grid_forget()
            self.btn_pietons_detection.grid()


    def open_webcam(self):
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        self.is_webcam = True
        self.canvas.config(bg="gray")
        self.btn_example1.grid()
        self.btn_pietons_detection.grid_forget()

    def switch_video_source(self):
        if self.vid:
            self.vid.release()
        if self.is_webcam:
            self.load_video()
        else:
            self.open_webcam()

    def toggle_play_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.config(text="Play")
        else:
            self.btn_pause.config(text="Pause")

    def take_screenshot(self):
        if self.vid is not None:
            # Create a postscript file and save the canvas content
            ps_file_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ps"
            self.canvas.postscript(file=ps_file_path, colormode='color')

            # Close the postscript file
            self.canvas.update()

            # Convert the postscript file to an image
            screenshot = Image.open(ps_file_path)
            screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            screenshot.save(screenshot_path, format="png")
            print(f"Screenshot saved at: {screenshot_path}")

            # Close the Image object before attempting to delete the file
            screenshot.close()

            # Introduce a delay before attempting to delete the file
            time.sleep(1)

            # Close and delete the postscript file
            os.remove(ps_file_path)

    def update(self):
        if self.vid is not None and not self.is_paused:
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                if self.vid and self.current_detection_method == "Haar" and not self.is_webcam:
                    print("haar vid")
                    self.haar_video(frame)

                elif self.vid and self.current_detection_method == "Yolo" and not self.is_webcam:
                    print("Yolo vid")
                    self.yolo_video(frame)

                if self.is_webcam and self.current_detection_method == "Haar":
                    print("haar is_webcam")

                elif self.is_webcam and self.current_detection_method == "Yolo":
                    print("Yolo is_webcam")
                    self.yolo_webcam(frame)

                elif self.current_detection_method == "Stop":
                    pass

                if not self.is_paused:
                    self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            else:
                self.canvas.config(bg="gray")
        self.root.after(100, self.update)

    def __del__(self):
        if self.vid is not None and self.vid.isOpened():
            self.vid.release()

# Create and run the application
root = tk.Tk()
app = VideoApp(root)
root.mainloop()
