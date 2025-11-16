import os
from typing import List
import cv2
import numpy as np
from matplotlib import pyplot as plt

VIDEO_PATH = "videoTracking/video_with_marker2.mp4"
NFEATURES = 200

# >>> NEW: saving config
NUM_MASK_SAMPLES = 200
DATASET_DIR = "U_Net_dataset2/ds3_regular_case/raw"
SAVE_DIR = "masks"
SAVE_ORIG_DIR = "imgs"
RANDOM_SEED = 42
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, SAVE_DIR), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR,SAVE_ORIG_DIR), exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
video_w, video_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('orb_features.mp4', fourcc, 30.0, (video_w,  video_h))
opening_square_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

# >>> NEW: preselect random frame indices to save
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames and total_frames > 0:
    k = min(NUM_MASK_SAMPLES, total_frames)
    sample_indices = set(rng.choice(total_frames, size=k, replace=False).tolist())
else:
    # Fallback if frame count is unknown: sample every Nth approx, then random offset
    approx_stride = 100  # adjust if you want more/less
    sample_indices = set()
    # we'll also fill this on-the-fly below if needed
# <<< NEW

def filter_by_area_range(mask, min_area=200, max_area=1500, connectivity=8):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = (areas >= min_area) & (areas <= max_area)
    keep[0] = False  # background

    filtered = np.where(keep[labels], 255, 0).astype(np.uint8)
    return filtered

def unsharp_mask(image, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(image, (3, 3), sigma)
    sharpened = cv2.addWeighted(src1=image, alpha=1.0 + strength, src2=blurred, beta=-strength, gamma=0)
    return sharpened

def draw_keypoints(img_color, keypoints):
    for kp in keypoints:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        cv2.circle(img_color, (x, y), radius=3, color=(0, 255, 0), thickness=1)
    return img_color

def main():
    frame_idx = 0
    # >>> NEW (fallback) prepare lazy sampling if frame count unknown
    need_lazy_sampling = (not total_frames) or (total_frames <= 0)
    saved_count = 0
    # <<< NEW

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error occurred.")
            break

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        orb_extractor = cv2.ORB_create(nfeatures=1000, edgeThreshold=0)
        blurred = cv2.medianBlur(img_gray, 5)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(blurred)
        _, image_binary = cv2.threshold(enhanced_image, 30, 255, cv2.THRESH_BINARY_INV)

        orb_keypoints = orb_extractor.detect(enhanced_image, None)
        mask = cv2.morphologyEx(image_binary, cv2.MORPH_OPEN, opening_square_ellipse)
        final_mask = filter_by_area_range(mask)
        final_mask[:, :200] = 0

        frame_show = draw_keypoints(cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR), orb_keypoints)
        out.write(frame_show)

        # Show if you want (optional)
        cv2.imshow("final_mask", final_mask)
        # cv2.imshow("binary", image_binary)

        # >>> NEW: save randomly selected masks
        do_save = False
        if total_frames and total_frames > 0:
            # preselected indices
            if frame_idx in sample_indices:
                do_save = True
        else:
            # lazy reservoir-like sampling without storing in RAM
            # save first NUM_MASK_SAMPLES, then with decreasing probability replace a previous one
            if saved_count < NUM_MASK_SAMPLES:
                do_save = True
            else:
                # with probability NUM_MASK_SAMPLES / (frame_idx+1), overwrite a previous saved file
                if rng.random() < (NUM_MASK_SAMPLES / float(frame_idx + 1)):
                    # overwrite a random previously saved index
                    replace_idx = rng.integers(0, NUM_MASK_SAMPLES)
                    # delete old file with that slot if exists
                    old_path = os.path.join(SAVE_DIR, f"mask_{replace_idx:03d}.png")
                    if os.path.exists(old_path):
                        os.remove(old_path)
                    # mark we will save this frame into that slot
                    do_save = True
                    # reuse saved_count to store into chosen slot name below
                    saved_count = replace_idx  # will be incremented to replace_idx+1 below

        if do_save:
            base = f"f{frame_idx:06d}"
            mask_path = os.path.join(DATASET_DIR, SAVE_DIR,      f"{base}.png")
            orig_path = os.path.join(DATASET_DIR, SAVE_ORIG_DIR, f"{base}.jpg")
            # # choose filename
            # if total_frames and total_frames > 0:
            #     fname = os.path.join(SAVE_DIR, f"mask_f{frame_idx:06d}.png")
            # else:
            #     # numbered slots 0..NUM_MASK_SAMPLES-1 for lazy sampling
            #     fname = os.path.join(SAVE_DIR, f"mask_{saved_count:03d}.png")
            cv2.imwrite(mask_path, final_mask)  # final_mask is uint8 0/255
            cv2.imwrite(orig_path, frame)
            if not (total_frames and total_frames > 0):
                saved_count += 1

        # Quit key (optional)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
