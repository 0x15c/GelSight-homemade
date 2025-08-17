import numpy as np
import cv2 as cv
from scipy.fft import fft2, ifft2
import pyvista as pv

def poisson_reconstruct(p,q):

    # divergence = dp/dx + dq/dy
    dpdx = np.gradient(p, axis=1)
    dqdy = np.gradient(q, axis=0)
    f = dpdx + dqdy

    H, W = f.shape
    fy, fx = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing="ij")

    denom = (2*np.pi*1j*fx)**2 + (2*np.pi*1j*fy)**2
    denom[0,0] = 1  # avoid div/0 for DC component

    F = fft2(f)
    Z = np.real(ifft2(F / denom))

    # normalize depth
    # Z -= Z.min()
    # Z /= Z.max()
    return Z

num_bins = 64

bg = cv.imread('nomarker_ref.jpg').astype(np.int16)
img = cv.imread('test_data/sample_8.jpg').astype(np.int16)
im_diff = img - bg

H, W, _ = img.shape

b_l = (im_diff[:,:,0]+255)//(512//num_bins)
g_l = (im_diff[:,:,1]+255)//(512//num_bins)
r_l = (im_diff[:,:,2]+255)//(512//num_bins)

fancyTable=np.load('lookuptable.npy')

Grad_im = fancyTable[b_l, g_l, r_l,:]

GradX = Grad_im[:,:,0]
GradY = Grad_im[:,:,1]

y_idx, x_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
X = x_idx
Y = y_idx
Z = poisson_reconstruct(GradX, GradY)


points = np.stack((X,Y,Z),axis=-1).reshape(-1, 3)

cloud = pv.PolyData(points)
cloud.plot(point_size=5)
surf = cloud.delaunay_2d()
surf.plot(show_edges=True)

# Z_reg = (Z*255).astype(np.uint8)

# heatmap = cv.applyColorMap(Z_reg, cv.COLORMAP_JET)

# cv.imshow('heatmap',heatmap)

GradX_im = ((GradX - np.min(GradX))/(np.max(GradX)-np.min(GradX))*255).astype(np.uint8)
GradY_im = ((GradY - np.min(GradY))/(np.max(GradY)-np.min(GradY))*255).astype(np.uint8)

# display image in red and blue channel, red for Gy and blue for Gx
img_red = np.stack([np.zeros_like(GradY_im),np.zeros_like(GradY_im),GradY_im],axis=2)
img_blue = np.stack([GradX_im,np.zeros_like(GradX_im),np.zeros_like(GradX_im)],axis=2)

img = np.concat([img_red,img_blue],axis=0)

cv.imshow('img',img)
cv.waitKey(0)