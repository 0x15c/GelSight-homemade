import numpy as np
import cv2 as cv

num_bins = 64

bg = cv.imread('nomarker_ref.jpg').astype(np.int16)
img = cv.imread('test_data/reconstruct_4.jpg').astype(np.int16)
im_diff = img - bg

b_l = (im_diff[:,:,0]+255)//(512//num_bins)
g_l = (im_diff[:,:,1]+255)//(512//num_bins)
r_l = (im_diff[:,:,2]+255)//(512//num_bins)

fancyTable=np.load('lookuptable.npy')

Grad_im = fancyTable[b_l, g_l, r_l,:]

GradX = Grad_im[:,:,0]
GradY = Grad_im[:,:,1]

GradX_im = ((GradX - np.min(GradX))/(np.max(GradX)-np.min(GradX))*255).astype(np.uint8)
GradY_im = ((GradY - np.min(GradY))/(np.max(GradY)-np.min(GradY))*255).astype(np.uint8)

# display image in red and blue channel, red for Gy and blue for Gx
img_red = np.stack([np.zeros_like(GradY_im),np.zeros_like(GradY_im),GradY_im],axis=2)
img_blue = np.stack([GradX_im,np.zeros_like(GradX_im),np.zeros_like(GradX_im)],axis=2)

img = np.concat([img_red,img_blue],axis=0)

cv.imshow('img',img)
cv.waitKey(0)