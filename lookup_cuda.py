# gpu_poisson_lookup_o3d.py
import numpy as np
import cv2 as cv
import pandas as pd
import time
# CuPy optional
try:
    import cupy as cp
    from cupy.fft import fft2 as gpu_fft2, ifft2 as gpu_ifft2, fftfreq as gpu_fftfreq
    xp = cp
    use_gpu = True
except Exception as e:
    print("CuPy not available, falling back to NumPy (CPU). Error:", e)
    from scipy.fft import fft2 as cpu_fft2, ifft2 as cpu_ifft2, fftfreq as cpu_fftfreq
    xp = np
    use_gpu = False

# Open3D
import open3d as o3d


def poisson_reconstruct(p, q):
    """Poisson integration on xp backend"""
    H, W = p.shape
    dpdx = xp.gradient(p, axis=1)
    dqdy = xp.gradient(q, axis=0)
    f = dpdx + dqdy

    if use_gpu:
        fy = gpu_fftfreq(H)[:, None]
        fx = gpu_fftfreq(W)[None, :]
        F = gpu_fft2(f)
        denom = - (2 * xp.pi) ** 2 * (fx**2 + fy**2)
        denom[0, 0] = 1.0
        Z = xp.real(gpu_ifft2(F / denom))
    else:
        fy = cpu_fftfreq(H)[:, None]
        fx = cpu_fftfreq(W)[None, :]
        F = cpu_fft2(f)
        denom = - (2 * np.pi) ** 2 * (fx**2 + fy**2)
        denom[0, 0] = 1.0
        Z = np.real(cpu_ifft2(F / denom))

    # remove constant offset
    Z = Z - Z[0, 0]
    return Z


def main():
    num_bins = 64
    bg_cpu = cv.imread("calib_0828/crop/bg/bg.jpg").astype(np.int16)
    img_cpu = cv.imread("reconstruct0828/crop/8.jpg").astype(np.int16)
    im_diff_cpu = img_cpu - bg_cpu

    H, W, _ = img_cpu.shape

    if use_gpu:
        im_diff = cp.asarray(im_diff_cpu)
    else:
        im_diff = im_diff_cpu

    scale = 512 // num_bins if 512 // num_bins > 0 else 1
    b_l = (im_diff[:, :, 0] + 255) // scale
    g_l = (im_diff[:, :, 1] + 255) // scale
    r_l = (im_diff[:, :, 2] + 255) // scale

    b_l = xp.clip(b_l, 0, num_bins - 1).astype(xp.int32)
    g_l = xp.clip(g_l, 0, num_bins - 1).astype(xp.int32)
    r_l = xp.clip(r_l, 0, num_bins - 1).astype(xp.int32)

    # load LUT
    fancyTable_cpu = np.load("lut_0904.npy")
    fancyTable = cp.asarray(fancyTable_cpu) if use_gpu else fancyTable_cpu
    t0 = time.time()
    print("Lookup Table Loaded.")
    # lookup

    Grad_im = fancyTable[b_l, g_l, r_l, :]
    GradX = Grad_im[:, :, 0].astype(xp.float64)
    GradY = Grad_im[:, :, 1].astype(xp.float64)
    t1 = time.time()
    print(f"Lookup Process cost: {t1-t0}s")
    # reconstruct
    t2 = time.time()
    Z = poisson_reconstruct(GradX, GradY)
    print(f"Poisson Reconstruct Time: {t2-t1}s")

    y_idx = xp.arange(H)[:, None].astype(xp.float64)
    x_idx = xp.arange(W)[None, :].astype(xp.float64)
    X = xp.tile(x_idx, (H, 1))
    Y = xp.tile(y_idx, (1, W))

    if use_gpu:
        points = cp.asnumpy(xp.stack((X, Y, Z), axis=-1).reshape(-1, 3))
    else:
        points = xp.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    # --- Open3D visualization ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # estimate normals (optional, for better rendering)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    t = time.time() - t0
    print(f"From begin of table lookup to open3D window created cost: {t}s")
    o3d.visualization.draw_geometries([pcd])


def video():
    num_bins = 64
    video_path = "video/demo.mp4"
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    video_w, video_h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # camera settings
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, video_w)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, video_h)
    cap.set(cv.CAP_PROP_EXPOSURE, -6.4)
    cap.set(cv.CAP_PROP_BRIGHTNESS, 50)
    cap.set(cv.CAP_PROP_CONTRAST, 64)
    cap.set(cv.CAP_PROP_SATURATION, 50)
    cap.set(cv.CAP_PROP_HUE, 0)
    cap.set(cv.CAP_PROP_GAIN, 0)
    # crop settings
    cnt = [int(video_w/2),int(video_h/2)]
    crop_px = 280
    crop_py = 240
    crop_offset_x = -30
    crop_offset_y = 0
    cropped_limits = [[cnt[0]-crop_px+crop_offset_x,cnt[1]-crop_py+crop_offset_y],[cnt[0]+crop_px+crop_offset_x,cnt[1]+crop_py+crop_offset_y]]
    cropped_size = [2*crop_px, 2*crop_py]


    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        print("Video file opened successfully")
        frame_count = 0
        # --- create Open3D visualizer window ---
        vis = o3d.visualization.Visualizer()
        vis.create_window("Point Cloud Stream")
        geometry = o3d.geometry.PointCloud()
        vis.add_geometry(geometry)
        vis.reset_view_point(True)
        # ctr = vis.get_view_control()
        # load LUT
        fancyTable_cpu = np.load("lut_0904.npy")
        fancyTable = cp.asarray(fancyTable_cpu) if use_gpu else fancyTable_cpu
        print("Lookup Table Loaded.")
        W, H = cropped_size
        y_idx = xp.arange(H)[:, None].astype(xp.float64)
        x_idx = xp.arange(W)[None, :].astype(xp.float64)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error occurred.")
            break
        t0 = time.time()
        frame_cropped = frame[cropped_limits[0][1]:cropped_limits[1][1],cropped_limits[0][0]:cropped_limits[1][0]]
        if frame_count == 10: # waiting for stablization
            init_frame = frame_cropped
            # vis.reset_view_point(True)
        if frame_count >=10 :
            # --- Table lookup ---
            frame_diff = np.int16(frame_cropped) - np.int16(init_frame)
            if use_gpu:
                im_diff = cp.asarray(frame_diff)
            else:
                im_diff = frame_diff
            scale = 512 // num_bins if 512 // num_bins > 0 else 1
            # lookup
            b_l = (im_diff[:, :, 0] + 255) // scale
            g_l = (im_diff[:, :, 1] + 255) // scale
            r_l = (im_diff[:, :, 2] + 255) // scale
            b_l = xp.clip(b_l, 0, num_bins - 1).astype(xp.int32)
            g_l = xp.clip(g_l, 0, num_bins - 1).astype(xp.int32)
            r_l = xp.clip(r_l, 0, num_bins - 1).astype(xp.int32)
            Grad_im = fancyTable[b_l, g_l, r_l, :]
            GradX = Grad_im[:, :, 0].astype(xp.float64)
            GradY = Grad_im[:, :, 1].astype(xp.float64)
            # reconstruct
            Z = poisson_reconstruct(GradX, GradY)

            X = xp.tile(x_idx, (H, 1))
            Y = xp.tile(y_idx, (1, W))
            if use_gpu:
                points = cp.asnumpy(xp.stack((X, Y, Z), axis=-1).reshape(-1, 3))
            else:
                points = xp.stack((X, Y, Z), axis=-1).reshape(-1, 3)

            # update Open3D point cloud
            geometry.points = o3d.utility.Vector3dVector(points)
            vis.clear_geometries()
            vis.add_geometry(geometry, True)
            vis.update_renderer()
            vis.get_view_control().set_front([ 0, -0.97, 0.24 ])
            vis.get_view_control().set_up([ 0, 0.24, 0.97 ])
            vis.poll_events()
            
            # o3d.visualization.draw_geometries([geometry])
        frame_count+=1
        time.sleep(0.0001)
        frame_show = cv.flip(frame_cropped,flipCode=0)
        frame_show = cv.putText(frame_show,f'FPS = {1/(time.time()-t0)}',(0,25),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1,cv.LINE_AA)
        cv.imshow("original input",frame_show)
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    cap.release()
    vis.destroy_window()



if __name__ == "__main__":
    video()

'''
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 1299.0, 999.0, 27.308139280173652 ],
			"boundingbox_min" : [ 0.0, 0.0, -9.8793729006256115 ],
			"field_of_view" : 60.0,
			"front" : [ -0.016418874082577386, -0.97028873997977227, 0.24139217394589785 ],
			"lookat" : [ 649.5, 499.5, 8.7143831897740203 ],
			"up" : [ -0.0037924473439299995, 0.24148341502378576, 0.97039753586434541 ],
			"zoom" : 0.69999999999999996
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
'''