import cv2 as cv
import numpy as np
import scipy.interpolate
from sklearn.cluster import DBSCAN

# this function takes extracted cluster info as input, calculates centroids with intensity
def centroids_calc(cluster_array):
    result = np.zeros((0,2))
    intsty = np.zeros((0))
    for cluster in cluster_array:
        centroid = np.mean(cluster,axis=0)
        n_pts = cluster.shape[0]
        intensity = n_pts # there shall be some mapping...
        result = np.append(result,[centroid],axis=0)
        intsty = np.append(intsty,[intensity],axis=0)
    return result, intsty

'''
    a function takes DBSCAN results, returns a tuple (n, k, 2) where:
    n is the # of clusters
    k is the points count inside a cluster
    2 is the (x, y) coordinate
 '''
def dbscan_extractor(dbscan_result, points):
    labels = dbscan_result.labels_
    points = np.array(points)
    
    # Get unique cluster labels (excluding noise)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    
    if len(unique_labels) == 0:
        return []
    
    cluster_info = []
    
    for cluster_id in unique_labels:
        cluster_mask = (labels == cluster_id)
        cluster_points = points[cluster_mask]
        cluster_info.append(cluster_points)  # Just append the actual points
    
    return cluster_info



class Remove_marker:
    # takes ref image as input, 3 channels
    # constructs referance array
    # workflow: filter out markers
    def __init__(self,image,returnInterpolatedImage=False):
        grey_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        self.empty_mask = np.zeros_like(image[:,:,0])
        self.row, self.col = np.shape(self.empty_mask)
        # self.blurred = cv.GaussianBlur(image,(5,5),sigmaX=2,sigmaY=2)
        # split color channels
        self.b = image[:,:,0]
        self.g = image[:,:,1]
        self.r = image[:,:,2]

        # remove markers: using dense interpolation, b channel for example
        # image preparation: threshold
        rspace = np.linspace(0,self.row-1,self.row).astype(np.uint32)
        cspace = np.linspace(0,self.col-1,self.col).astype(np.uint32)
        cgrid, rgrid = np.meshgrid(cspace,rspace)

        self.mask = cv.adaptiveThreshold(grey_image,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,15,2).astype(bool)

        o_indices = np.where(self.mask==False)
        o_coordinates = np.stack(o_indices, axis=1)

        cluster_data = DBSCAN(eps=5, min_samples=35).fit(o_coordinates)
        clusters = dbscan_extractor(cluster_data, o_coordinates)
        centroids,_ = centroids_calc(clusters)
        # self.b_new = scipy.interpolate.griddata(b_coordinates,b_val,(rgrid,cgrid),method='cubic').astype(np.uint8)
        canvas = np.ones_like(image)
        for item in centroids:
            r, c = int(item[1]), int(item[0])
            # mask can be any shape, choose one match the marker shape
            # cv.circle(canvas, (r,c), 8,(0,0,0),-1)
            cv.ellipse(canvas, (r,c), (10,7),0,0,360,(0,0,0),-1)
        self.o_mask = canvas[:,:,0].astype(bool)

        o_indices = np.where(self.o_mask)
        o_coordinates = np.stack(o_indices, axis=1)

        if returnInterpolatedImage:
            # conduct linear interpolation to rebuild image
            # @TODO: using dense interpolation data may be time-consuming, try Monte Carlo instead.
            b_val = self.b[self.o_mask]
            self.b_new = scipy.interpolate.griddata(o_coordinates,b_val,(rgrid,cgrid),method='linear').astype(np.uint8)
            g_val = self.g[self.o_mask]
            self.g_new = scipy.interpolate.griddata(o_coordinates,g_val,(rgrid,cgrid),method='linear').astype(np.uint8)
            r_val = self.r[self.o_mask]
            self.r_new = scipy.interpolate.griddata(o_coordinates,r_val,(rgrid,cgrid),method='linear').astype(np.uint8)

            # masks
            self.mask = self.mask.astype(np.uint8)*255 # this mask is derived by adaptive thresholding
            self.mask_circle = (canvas*255).astype(np.uint8) # this mask is used for making holes
            
            # returns image
            self.bgr = np.stack((self.b_new,self.g_new,self.r_new),axis=2)
            self.bgr = cv.GaussianBlur(self.bgr,(5,5),5,None,5)
        
# class Remove_marker_with_ball(Remove_marker):
#     def __init__(self, image):
#         # first, detect the location of ball

#         super().__init__(image, returnInterpolatedImage=False)
def diff_image(img_ref, img):
    img_ref = img_ref.astype(np.int16)
    img = img.astype(np.int16)
    img_diff = img - img_ref
    max, min = np.max(img_diff), np.min(img_diff)
    img_diff_visible = ((img_diff - min)/(max - min)*255).astype(np.uint8)

    return img_diff_visible
def find_center_manually(img_ball,init_center=None,init_radius=None):
    key=-1
    if init_center is None:
        c=[img_ball.shape[1]//2,img_ball.shape[0]//2]
    else:
        c=init_center
    if init_radius is None:
        r=10
    else:
        r=init_radius
    
    while key!=27: #ESC
        im2show = cv.circle(np.array(img_ball),c,r,(0,255,0),2) # creates a new instance of img_ball
        cv.imshow('contact image manually adjust',im2show)
        key=cv.waitKey(0)
        if key == 119: # W
            c[1] -= 1
        elif key == 115: # S
            c[1] += 1
        elif key == 97: # A
            c[0] -= 1
        elif key == 100: # D
            c[0] += 1
        elif key == 109: # M
            r += 1
        elif key == 110: # N
            r -= 1
    return c, r
def generate_ball_mask(img_ball, center, radius):
    canvas = np.zeros((img_ball.shape[0],img_ball.shape[1]))
    mask = cv.circle(canvas,center,radius,1,-1)
    mask = mask.astype(np.uint8)*255
    return mask

class Img_preprocess:
    def __init__(self, im_ref, im_ball):
        im_diff = diff_image(im_ref, im_ball) # generate the difference image
        self.c,self.r = find_center_manually(im_diff) # manually find center and radius
        ball_mask = generate_ball_mask(im_diff, self.c, self.r) # use center and radius to generate ball mask
        grey_im = cv.cvtColor(im_diff,cv.COLOR_BGR2GRAY) 
        masked = cv.bitwise_and(grey_im,grey_im,mask=ball_mask) # generate masked img
        _,self.mask = cv.threshold(masked,115,255,cv.THRESH_BINARY) # generate the final mask
        self.masked_img = cv.bitwise_and(im_ball,im_ball,mask=self.mask)

class Calib_param:
    def __init__(self, BallRad, mm2Pixel):
        self.ballrad = BallRad
        self.mm2pixel = mm2Pixel
        self.ballradPix = BallRad / mm2Pixel

class Gradient():

    def __init__(self, center, Radius, image, mask=None):
        # split b, g, r channel of image
        b = image[:,:,0]
        g = image[:,:,1]
        r = image[:,:,2]

        # if there exists mask, first extracts the masked coordinates
        if mask is None:
            pass
        # @TODO: need implementation here
        else:
            mask_binary = mask.astype(bool)
            self.pixCoord = np.stack(np.where(mask_binary==True),axis=1)
            self.surfNorm(self.pixCoord,center,Radius)
            pass


    def surfNorm(self, pixCoord, center, Radius):
        pixCoord[:,0] = pixCoord[:,0]-center[1]
        pixCoord[:,1] = pixCoord[:,1]-center[0]
        grad = np.zeros_like(pixCoord)
        for i in pixCoord:
            gi = self.G(i[0],i[1],Radius)
            grad[i]=gi
            pass



    def G(self,x,y,R): # Grad 2D
        r = lambda x,y: np.sqrt(x**2 + y**2)
        Dx = lambda x,y,R: -x/np.sqrt(R**2-r(x,y)**2)
        Dy = lambda x,y,R: -y/np.sqrt(R**2-r(x,y)**2)
        return np.array([Dx(x,y,R),Dy(x,y,R)])
    def N(self,x,y,R): # Norm 3D
        return np.array([x/R, y/R, np.sqrt(R**2-x**2-y**2)/R])



if __name__ == '__main__':
    im = cv.imread('nomarker_ref.jpg')
    im_ball = cv.imread('test_data/sample_8.jpg')
    img = Img_preprocess(im, im_ball)
    Grad = Gradient(img.c,img.r,im_ball,img.mask)
    cv.imshow('mask2',img.masked_img)
    cv.waitKey(0)