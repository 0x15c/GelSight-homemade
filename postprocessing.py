import os
from pathlib import Path
import cv2
import numpy as np

# === INPUT DATASET ===
IN_IMG_DIR = Path("ds3_regular_case/cropped/imgs")            # your input image crops
IN_MSK_DIR = Path("ds3_regular_case/cropped/masks")      # your input mask crops

# === OUTPUT FIXED DATASET ===
OUT_IMG_DIR = Path("ds3_regular_case/data_train/imgs")
OUT_MSK_DIR = Path("ds3_regular_case/data_train/masks")

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_MSK_DIR.mkdir(parents=True, exist_ok=True)

# === PARAMETERS ===
MIN_W = 300         # minimum width after tiling
MIN_H = 300         # minimum height after tiling
TARGET_W = 512      # final output width
TARGET_H = 512      # final output height

# -------------------------------------------------------------
# TILE IMAGE / MASK UNTIL IT REACHES MINIMUM SIZE
# -------------------------------------------------------------
def tile_until_min_size(img, min_w=300, min_h=300):
    """
    Repeat an image with itself horizontally & vertically until
    width >= min_w and height >= min_h.
    Works for both color (H,W,3) and grayscale (H,W).
    """
    h, w = img.shape[:2]

    # how many repetitions needed
    rep_w = max(1, (min_w // w) + 1 if w < min_w else 1)
    rep_h = max(1, (min_h // h) + 1 if h < min_h else 1)

    # tile
    if img.ndim == 3:   # color
        tiled = np.tile(img, (rep_h, rep_w, 1))
    else:               # grayscale
        tiled = np.tile(img, (rep_h, rep_w))

    return tiled


# -------------------------------------------------------------
# RESIZE TO FIXED 512x512 (non-uniform scaling allowed)
# -------------------------------------------------------------
def resize_fixed(img, target_h, target_w, is_mask=False):
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(img, (target_w, target_h), interpolation=interp)


# -------------------------------------------------------------
# MAIN PROCESSING LOOP
# -------------------------------------------------------------
def main():
    img_files = sorted(IN_IMG_DIR.glob("*.jpg"))
    print(f"Found {len(img_files)} input images.")

    kept = 0
    missing_masks = 0
    read_fail = 0

    for img_path in img_files:
        base = img_path.stem
        msk_path = IN_MSK_DIR / f"{base}.png"

        if not msk_path.exists():
            print(f"[WARN] Missing mask for {base}, skipping.")
            missing_masks += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)

        if img is None or msk is None:
            print(f"[WARN] Failed reading {base}, skipping.")
            read_fail += 1
            continue

        # -----------------------------------------------------
        # 1) TILE IMAGE AND MASK UNTIL >= 300x300
        # -----------------------------------------------------
        img_tiled = tile_until_min_size(img, min_w=MIN_W, min_h=MIN_H)
        msk_tiled = tile_until_min_size(msk, min_w=MIN_W, min_h=MIN_H)

        # -----------------------------------------------------
        # 2) RESIZE TO FINAL 512x512
        # -----------------------------------------------------
        img_fixed = resize_fixed(img_tiled, TARGET_H, TARGET_W, is_mask=False)
        msk_fixed = resize_fixed(msk_tiled, TARGET_H, TARGET_W, is_mask=True)

        # -----------------------------------------------------
        # 3) RE-BINARIZE MASK
        # -----------------------------------------------------
        _, msk_fixed = cv2.threshold(msk_fixed, 127, 255, cv2.THRESH_BINARY)

        # -----------------------------------------------------
        # 4) SAVE OUTPUT
        # -----------------------------------------------------
        cv2.imwrite(str(OUT_IMG_DIR / f"{base}.jpg"), img_fixed)
        cv2.imwrite(str(OUT_MSK_DIR / f"{base}.png"), msk_fixed)

        kept += 1

    print("\n=== DONE ===")
    print(f"Kept pairs        : {kept}")
    print(f"Missing masks     : {missing_masks}")
    print(f"Read failures     : {read_fail}")


if __name__ == "__main__":
    main()
