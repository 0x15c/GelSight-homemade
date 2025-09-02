import cv2 as cv
import numpy as np
import glob

bg = cv.imread("reconstruct0828/1.jpg")
w = bg.shape[1]
h = bg.shape[0]

# crop settings
cnt = [int(w/2),int(h/2)]
crop_px = 650
crop_py = 500
crop_offset_x = -50
crop_offset_y = 0
cropped_limits = [[cnt[0]-crop_px+crop_offset_x,cnt[1]-crop_py+crop_offset_y],[cnt[0]+crop_px+crop_offset_x,cnt[1]+crop_py+crop_offset_y]]
cropped_size = [2*crop_px, 2*crop_py]

img_list = sorted(glob.glob("reconstruct0828/*.jpg"))
for idx, i in enumerate(img_list):
    img = cv.imread(i)
    cv.imwrite(filename=f"reconstruct0828/crop/{idx}.jpg",img = img[cropped_limits[0][1]:cropped_limits[1][1],cropped_limits[0][0]:cropped_limits[1][0]])
    pass