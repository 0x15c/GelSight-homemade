import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cap = cv.VideoCapture(1)

cap.set(cv.CAP_PROP_EXPOSURE, -6.5)
cap.set(cv.CAP_PROP_BRIGHTNESS, 64)
cap.set(cv.CAP_PROP_CONTRAST, 100)
cap.set(cv.CAP_PROP_SATURATION, 64)
cap.set(cv.CAP_PROP_HUE, 0)
cap.set(cv.CAP_PROP_GAIN, 2.0)
plt.ion()
fig, ax = plt.subplots()
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
    color = ('b','g','r')
    ax.clear()
    for i,col in enumerate(color):
        histr = cv.calcHist([img],[i],None,[256],[0,256])   
        ax.plot(histr,color = col)
        plt.xlim([0,256])
        plt.pause(0.001)
    plt.show()
    key = cv.waitKey(1)
    if key & 0xFF == ord('q'):
        break