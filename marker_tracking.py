import cv2
import numpy as np
import time
video_path = "videoTracking/video_with_marker.mp4"
cap = cv2.VideoCapture(video_path)
# Get frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))*2
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (frame_width, frame_height))
# obtain a kernel
opening_square_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
else:
    print("Video file opened successfully. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error occurred.")
        break
    # 1. Convert the image into grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, image_binary = cv2.threshold(frame_gray, 50, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('binary',cv2.cvtColor(image_binary,cv2.COLOR_GRAY2BGR))
    opening_with_ellipse_kernel = cv2.morphologyEx(image_binary, cv2.MORPH_OPEN, opening_square_ellipse)
    cv2.imshow('opening',cv2.cvtColor(opening_with_ellipse_kernel,cv2.COLOR_GRAY2BGR))
    cv2.imshow('gray',frame_gray)
    # concatenate along the row
    output_frame = np.concatenate((cv2.cvtColor(opening_with_ellipse_kernel,cv2.COLOR_GRAY2BGR),frame),axis=0)
    out.write(output_frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()