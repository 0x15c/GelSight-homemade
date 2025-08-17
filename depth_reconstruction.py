import numpy as np
import cv2 as cv
from scipy.fft import fft2, ifft2
import pyvista as pv

class depth():
    def __init__(self, IMG_BG_PATH, LUT_PATH, NUM_BINS):
        self.bg = cv.imread(IMG_BG_PATH).astype(np.int16)
        self.H, self.W, _ = self.bg.shape
        self.lut = np.load(LUT_PATH)
        self.num_bins = NUM_BINS
        self.b_l = np.zeros_like(self.bg)
        self.g_l = np.zeros_like(self.bg)
        self.r_l = np.zeros_like(self.bg)
        self.Grad_img = np.zeros([self.H,self.W,2])
    def poisson_reconstruct(self, p,q):

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
    def get_depth_update(self, img):
        img_diff = img.astype(np.int16) - self.bg
        self.b_l = (img_diff[:,:,0]+255)//(512//self.num_bins)
        self.g_l = (img_diff[:,:,1]+255)//(512//self.num_bins)
        self.r_l = (img_diff[:,:,2]+255)//(512//self.num_bins)
        self.Grad_img = self.lut[self.b_l, self.g_l, self.r_l,:]
        self.depth = self.poisson_reconstruct(self.Grad_img[:,:,0],self.Grad_img[:,:,1])

if __name__ == '__main__':
    # shall see the usage
    my_depth = depth('nomarker_ref.jpg', 'lookuptable.npy', 64)
    img = cv.imread('test_data/reconstruct_8.jpg')
    my_depth.get_depth_update(img) # always use this to refresh
    y_idx, x_idx = np.meshgrid(np.arange(my_depth.H), np.arange(my_depth.W), indexing='ij')
    X = x_idx
    Y = y_idx
    Z = my_depth.depth
    points = np.stack((X,Y,Z),axis=-1).reshape(-1, 3)
    cloud = pv.PolyData(points)
    cloud.plot(point_size=5)
    surf = cloud.delaunay_2d()
    surf.plot(show_edges=True)
