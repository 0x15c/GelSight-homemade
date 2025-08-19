import numpy as np
import cv2 as cv
import time
import depth_reconstruction

cap = cv.VideoCapture(0)
fps = cap.get(cv.CAP_PROP_FPS)
video_w, video_h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# crop settings
cnt = [int(video_w/2),int(video_h/2)]
crop_px = 160
crop_py = 160
crop_offset_x = 0
crop_offset_y = 0
cropped_limits = [[cnt[0]-crop_px+crop_offset_x,cnt[1]-crop_py+crop_offset_y],[cnt[0]+crop_px+crop_offset_x,cnt[1]+crop_py+crop_offset_y]]
cropped_size = [2*crop_px, 2*crop_py]

# camera settings
cap.set(cv.CAP_PROP_FRAME_WIDTH, video_w)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, video_h)
cap.set(cv.CAP_PROP_EXPOSURE, -5)
cap.set(cv.CAP_PROP_BRIGHTNESS, 50)
cap.set(cv.CAP_PROP_CONTRAST, 64)
cap.set(cv.CAP_PROP_SATURATION, 50)
cap.set(cv.CAP_PROP_HUE, 0)
cap.set(cv.CAP_PROP_GAIN, 0)
# CAP_PROP_BRIGHTNESS: 0.0
# CAP_PROP_CONTRAST: 32.0
# CAP_PROP_SATURATION: 60.0
# CAP_PROP_HUE: 0.0
# CAP_PROP_GAIN: 0.0

# start of the main loop
key = -1
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame.")
        break
    time_start = time.time()
    frame_cropped = frame[cropped_limits[0][1]:cropped_limits[1][1],cropped_limits[0][0]:cropped_limits[1][0]]

    my_depth = depth_reconstruction.depth('calib_08182025/background.jpg', 'lut_0818.npy', 64)
    my_depth.get_depth_update(frame)
    y_idx, x_idx = np.meshgrid(np.arange(my_depth.H), np.arange(my_depth.W), indexing='ij')
    X = x_idx
    Y = y_idx
    Z = my_depth.depth
    Z_reg = (((Z-Z.min())/Z.max())*255).astype(np.uint8)
    heatmap = cv.applyColorMap(Z_reg,cv.COLORMAP_TURBO)
    cv.imshow('Regulated depth map',heatmap)
    cv.imshow('frame',frame_cropped)
    # if cv.waitKey(1) & 0xFF == ord('q'):
    #     break
    
    key = cv.waitKey(1)
    if key & 0xFF == ord('c'):
        cv.imwrite(filename=f'cropped_{time.time()}.jpg',img=frame)
        continue
    elif key & 0xFF == ord('q'):
        break





# ------------------CAMERA PARAMETER DETECTION SCRIPT------------------
# cap = cv.VideoCapture(0)

# # List of properties to check
# properties = {
#     "CAP_PROP_FRAME_WIDTH": cv.CAP_PROP_FRAME_WIDTH,
#     "CAP_PROP_FRAME_HEIGHT": cv.CAP_PROP_FRAME_HEIGHT,
#     "CAP_PROP_FPS": cv.CAP_PROP_FPS,
#     "CAP_PROP_BRIGHTNESS": cv.CAP_PROP_BRIGHTNESS,
#     "CAP_PROP_CONTRAST": cv.CAP_PROP_CONTRAST,
#     "CAP_PROP_SATURATION": cv.CAP_PROP_SATURATION,
#     "CAP_PROP_HUE": cv.CAP_PROP_HUE,
#     "CAP_PROP_GAIN": cv.CAP_PROP_GAIN,
#     "CAP_PROP_EXPOSURE": cv.CAP_PROP_EXPOSURE,
#     "CAP_PROP_FOCUS": cv.CAP_PROP_FOCUS,
#     "CAP_PROP_AUTOFOCUS": cv.CAP_PROP_AUTOFOCUS,
# }

# # Print supported properties
# for prop_name, prop_id in properties.items():
#     value = cap.get(prop_id)
#     if value != -1:  # -1 means the property is not supported
#         print(f"{prop_name}: {value}")
#     else:
#         print(f"{prop_name}: Not supported")

# cap.release()