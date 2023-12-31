# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
#
# ORIGINAL_WINDOW_TITLE = 'Original'
# FIRST_FRAME_WINDOW_TITLE = 'First Frame'
# DIFFERENCE_WINDOW_TITLE = 'Difference'
#
#
# canvas = None
# drawing = False # true if mouse is pressed
#
# #Retrieve first frame
# def initialize_camera(cap):
#     _, frame = cap.read()
#     return frame
#
#
# # mouse callback function
# def mouse_draw_rect(event,x,y,flags, params):
#     global drawing, canvas
#
#     if drawing:
#         canvas = params[0].copy()
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         params.append((x,y)) #Save first point
#
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing:
#             cv2.rectangle(canvas, params[1],(x,y),(0,255,0),2)
#
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         params.append((x,y)) #Save second point
#         cv2.rectangle(canvas,params[1],params[2],(0,255,0),2)
#
#
# def select_roi(frame):
#     global canvas
#     canvas = frame.copy()
#     params = [frame]
#     ROI_SELECTION_WINDOW = 'Select ROI'
#     cv2.namedWindow(ROI_SELECTION_WINDOW)
#     cv2.setMouseCallback(ROI_SELECTION_WINDOW, mouse_draw_rect, params)
#     roi_selected = False
#     while True:
#         cv2.imshow(ROI_SELECTION_WINDOW, canvas)
#         key = cv2.waitKey(10)
#
#         #Press Enter to break the loop
#         if key == 13:
#             break;
#
#
#     cv2.destroyWindow(ROI_SELECTION_WINDOW)
#     roi_selected = (3 == len(params))
#
#     if roi_selected:
#         p1 = params[1]
#         p2 = params[2]
#         if (p1[0] == p2[0]) and (p1[1] == p2[1]):
#             roi_selected = False
#
#     #Use whole frame if ROI has not been selected
#     if not roi_selected:
#         print('ROI Not Selected. Using Full Frame')
#         p1 = (0,0)
#         p2 = (frame.shape[1] - 1, frame.shape[0] -1)
#
#
#     return roi_selected, p1, p2
#
#
#
#
# if __name__ == '__main__':
#
#     cap = cv2.VideoCapture(0)
#
#     #Grab first frame
#     first_frame = initialize_camera(cap)
#
#     #Select ROI for processing. Hit Enter after drawing the rectangle to finalize selection
#     roi_selected, point1, point2 = select_roi(first_frame)
#
#     #Grab ROI of first frame
#     first_frame_roi = first_frame[point1[1]:point2[1], point1[0]:point2[0], :]
#
#     #An empty image of full size just for visualization of difference
#     difference_image_canvas = np.zeros_like(first_frame)
#
#     while cap.isOpened():
#
#         ret, frame = cap.read()
#
#         if ret:
#
#             #ROI of current frame
#             roi = frame[point1[1]:point2[1], point1[0]:point2[0], :]
#
#             difference = cv2.absdiff(first_frame_roi, roi)
#             difference = cv2.GaussianBlur(difference, (3, 3), 0)
#
#             _, difference = cv2.threshold(difference, 18, 255, cv2.THRESH_BINARY)
#
#
#             #Overlay computed difference image onto the whole image for visualization
#             difference_image_canvas[point1[1]:point2[1], point1[0]:point2[0], :] = difference.copy()
#
#
#             cv2.imshow(FIRST_FRAME_WINDOW_TITLE, first_frame)
#             cv2.imshow(ORIGINAL_WINDOW_TITLE, frame)
#             cv2.imshow(DIFFERENCE_WINDOW_TITLE, difference_image_canvas)
#
#
#             key = cv2.waitKey(30) & 0xff
#             if key == 27:
#                 break
#         else:
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()


import cv2, sys

cap = cv2.videocapture(sys.argv[1])
cv2.namedwindow('frame', cv2.window_normal)

# our roi, defined by two points
p1, p2 = none, none
state = 0


# called every time a mouse event happen
def on_mouse(event, x, y, flags, userdata):
    global state, p1, p2

    # left click
    if event == cv2.event_lbuttonup:
        # select first point
        if state == 0:
            p1 = (x, y)
            state += 1
        # select second point
        elif state == 1:
            p2 = (x, y)
            state += 1
    # right click (erase current roi)
    if event == cv2.event_rbuttonup:
        p1, p2 = none, none
        state = 0


# register the mouse callback
cv2.setmousecallback('frame', on_mouse)

while cap.isopened():
    val, frame = cap.read()

    # if a roi is selected, draw it
    if state > 1:
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 10)
    # show image
    cv2.imshow('frame', frame)

    # let opencv manage window events
    key = cv2.waitkey(50)
    # if escape key pressed, stop
    if key == 27:
        cap.release()