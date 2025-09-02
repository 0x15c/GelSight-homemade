import numpy as np
import cv2 as cv

cap = cv.VideoCapture(1)

while(True):
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame.")
        break
    img = frame
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]

    img_zero = np.zeros_like(b)

    img_B = np.stack([b,img_zero,img_zero],axis=2)
    img_G = np.stack([img_zero,g,img_zero],axis=2)
    img_R = np.stack([img_zero,img_zero,r],axis=2)

    bgr_img_r1 = np.concat([img,img_B],axis = 1)
    bgr_img_r2 = np.concat([img_G,img_R],axis = 1)
    
    bgr_img = np.concat([bgr_img_r1,bgr_img_r2],axis = 0)

    cv.imshow('split channel image',bgr_img)
    key = cv.waitKey(1)
    if key & 0xFF == ord('q'):
        break