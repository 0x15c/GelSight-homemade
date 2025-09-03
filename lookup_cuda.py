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
    fancyTable_cpu = np.load("lut_0828.npy")
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
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    t = time.time() - t0
    print(f"From begin of table lookup to open3D window created cost: {t}s")
    o3d.visualization.draw_geometries([pcd])


def video():
    num_bins = 64
    video_path = "demo_video/demo2.mp4"
    cap = cv.VideoCapture(video_path)
    # crop settings
    video_w, video_h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    cnt = [int(video_w/2),int(video_h/2)]
    crop_px = 400
    crop_py = 300
    crop_offset_x = -80
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
        # load LUT
        fancyTable_cpu = np.load("lut_0828.npy")
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
        if frame_count == 0:
            init_frame = frame_cropped
            vis.reset_view_point(True)
        if frame_count >=1 :
            # --- Table lookup ---
            frame_diff = frame_cropped - init_frame
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
            vis.add_geometry(geometry)
            vis.poll_events()
            vis.update_renderer()
            # o3d.visualization.draw_geometries([geometry])
        frame_count+=1
        time.sleep(0.0001)
        frame_cropped = cv.putText(frame_cropped,f'FPS = {1/(time.time()-t0)}',(50,50),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,255),1,cv.LINE_AA)
        cv.imshow("original input",frame_cropped)
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    cap.release()
    vis.destroy_window()



if __name__ == "__main__":
    video()

'''
import numpy as np
import open3d as o3d
import cv2 as cv

# --- create Open3D visualizer window ---
vis = o3d.visualization.Visualizer()
vis.create_window("Point Cloud Stream", width=960, height=540)
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

# open video
cap = cv.VideoCapture("your_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- replace this with your reconstruction pipeline ---
    H, W, _ = frame.shape
    Z = np.random.rand(H, W) * 50  # fake depth for demo
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    # update Open3D point cloud
    pcd.points = o3d.utility.Vector3dVector(points)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

cap.release()
vis.destroy_window()

'''