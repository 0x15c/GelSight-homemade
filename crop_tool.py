import os
import cv2
import glob
from pathlib import Path

# ========= CONFIG =========
IMG_DIR       = "ds3_regular_case/raw/imgs"           # f00xxxx.jpg
MSK_DIR       = "ds3_regular_case/raw/masks"     # f00xxxx.png
OUT_IMG_DIR   = "ds3_regular_case/cropped/imgs"
OUT_MSK_DIR   = "ds3_regular_case/cropped/masks"

IMG_EXT = ".jpg"
MSK_EXT = ".png"

# Display canvas limits (dual-panel fits inside this)
DISPLAY_MAX_W = 2000
DISPLAY_MAX_H = 2000

# Panel spacing in pixels
PANEL_GAP = 12

# Hotkeys:
#  s = save rects, n/Enter = next, p = previous, u = undo last, c = clear, q/Esc = quit
# ==========================

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MSK_DIR, exist_ok=True)

def is_enter_key(k: int) -> bool:
    """
    Robust Enter detection across platforms & keyboards.
    Handles Return (13), LineFeed (10), and some extended codes.
    """
    if k in (13, 10, 343):            # common cases (343 seen on some setups)
        return True
    kk = k & 0xFF                     # also check masked version
    return kk in (13, 10)

def list_common_basenames(img_dir, msk_dir, img_ext, msk_ext):
    imgs = {Path(p).stem for p in glob.glob(str(Path(img_dir) / f"*{img_ext}"))}
    msks = {Path(p).stem for p in glob.glob(str(Path(msk_dir) / f"*{msk_ext}"))}
    return sorted(imgs & msks)

def compute_shared_scale(w, h, max_w, max_h, gap):
    """
    For side-by-side panels of size (w,h) each, find a single scale so that
    the combined canvas (2*w*scale + gap, h*scale) fits inside (max_w, max_h).
    """
    if w <= 0 or h <= 0:
        return 1.0
    sx = max_w / float(2 * w + gap)
    sy = max_h / float(h)
    scale = min(1.0, sx, sy)
    return scale

def clamp_rect(x0, y0, x1, y1, w, h):
    x0, x1 = max(0, min(x0, w-1)), max(0, min(x1, w-1))
    y0, y1 = max(0, min(y0, h-1)), max(0, min(y1, h-1))
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

def save_crops(base, rects_fullres, img, msk, out_img_dir, out_msk_dir, img_ext, msk_ext):
    """rects_fullres: list of (x0,y0,x1,y1) in full-res coordinates"""
    # find last used suffix
    existing = sorted(Path(out_img_dir).glob(f"{base}_c*.{img_ext.lstrip('.')}"))
    start_idx = 0
    if existing:
        last = existing[-1].stem  # e.g., f001234_c07
        try:
            start_idx = int(last.split("_c")[-1])
        except Exception:
            start_idx = 0

    saved = 0
    idx = start_idx
    for (x0, y0, x1, y1) in rects_fullres:
        crop_img = img[y0:y1, x0:x1]
        crop_msk = msk[y0:y1, x0:x1]
        if crop_img.size == 0 or crop_msk.size == 0:
            continue
        idx += 1
        suffix = f"_c{idx:02d}"
        cv2.imwrite(str(Path(out_img_dir) / f"{base}{suffix}{img_ext}"), crop_img)
        cv2.imwrite(str(Path(out_msk_dir) / f"{base}{suffix}{msk_ext}"), crop_msk)
        saved += 1
    return saved

class DualPanelCropper:
    """
    Side-by-side display:
    [  image_panel (left)  |gap|  mask_panel (right)  ]

    - Draw on either; rect mirrored on both.
    - Rects stored in FULL-RES coordinates.
    """
    def __init__(self, img_bgr, msk_gray, base_name):
        self.img = img_bgr
        self.msk = msk_gray
        self.base = base_name

        # Ensure same size
        if self.img.shape[:2] != self.msk.shape[:2]:
            self.msk = cv2.resize(self.msk, (self.img.shape[1], self.img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        self.h, self.w = self.img.shape[:2]

        # Compute a single scale for both panels
        self.scale = compute_shared_scale(self.w, self.h, DISPLAY_MAX_W, DISPLAY_MAX_H, PANEL_GAP)
        disp_w = int(round(self.w * self.scale))
        disp_h = int(round(self.h * self.scale))
        self.panel_w = disp_w
        self.panel_h = disp_h
        self.split_x = self.panel_w + PANEL_GAP  # x coordinate where right panel starts (in canvas coords)

        # Prepare display copies
        self.left_disp  = cv2.resize(self.img, (self.panel_w, self.panel_h), interpolation=cv2.INTER_AREA)
        # colorize mask for right panel (so it’s easy to see)
        msk_vis = cv2.cvtColor(self.msk, cv2.COLOR_GRAY2BGR)
        self.right_disp = cv2.resize(msk_vis, (self.panel_w, self.panel_h), interpolation=cv2.INTER_NEAREST)

        # Canvas to draw
        self.canvas = self._compose_canvas()

        # Rectangles are kept in FULL-RES coords
        self.rects_full = []
        self.drawing = False
        self.pt0_full = None  # full-res start point

        self.window = "Crop: left=image | right=mask  (Enter=save+next, s=save, u=undo, c=clear, n=next, p=prev, q=quit)"
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self.on_mouse)

    def _compose_canvas(self):
        H = self.panel_h
        W = self.panel_w * 2 + PANEL_GAP
        canvas = 255 * np.ones((H, W, 3), dtype=np.uint8)
        canvas[:, 0:self.panel_w] = self.left_disp
        canvas[:, self.split_x:self.split_x+self.panel_w] = self.right_disp
        # headers
        cv2.putText(canvas, "IMAGE", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30,200,30), 2, cv2.LINE_AA)
        cv2.putText(canvas, "MASK",  (self.split_x+10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30,200,30), 2, cv2.LINE_AA)
        return canvas

    def _full_to_disp(self, x, y):
        """Map full-res (x,y) -> display panel coords (sx, sy)."""
        sx = int(round(x * self.scale))
        sy = int(round(y * self.scale))
        return sx, sy

    def _disp_to_full(self, x, y, on_right_panel):
        """Map canvas coords (x,y) to full-res (X,Y), accounting for panel offset."""
        if on_right_panel:
            x_local = x - self.split_x
        else:
            x_local = x
        x_local = max(0, min(x_local, self.panel_w-1))
        y_local = max(0, min(y, self.panel_h-1))
        inv = 1.0 / self.scale
        X = int(round(x_local * inv))
        Y = int(round(y_local * inv))
        # clamp
        X = max(0, min(X, self.w-1))
        Y = max(0, min(Y, self.h-1))
        return X, Y

    def on_mouse(self, event, x, y, flags, param):
        on_right = (x >= self.split_x)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.pt0_full = self._disp_to_full(x, y, on_right)
            self.show(preview=(self.pt0_full, self._disp_to_full(x, y, on_right)))
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.show(preview=(self.pt0_full, self._disp_to_full(x, y, on_right)))
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            pt1 = self._disp_to_full(x, y, on_right)
            x0, y0 = self.pt0_full
            x1, y1 = pt1
            x0, y0, x1, y1 = clamp_rect(x0, y0, x1, y1, self.w, self.h)
            if abs(x1 - x0) >= 4 and abs(y1 - y0) >= 4:
                self.rects_full.append((x0, y0, x1, y1))
            self.show()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # quick undo on right click
            if self.rects_full:
                self.rects_full.pop()
                self.show()

    def draw_rects_on_canvas(self, canvas, rects):
        # draw on LEFT
        for (x0, y0, x1, y1) in rects:
            sx0, sy0 = self._full_to_disp(x0, y0)
            sx1, sy1 = self._full_to_disp(x1, y1)
            cv2.rectangle(canvas, (sx0, sy0), (sx1, sy1), (0, 255, 0), 2)
        # draw on RIGHT (offset by split_x)
        for (x0, y0, x1, y1) in rects:
            sx0, sy0 = self._full_to_disp(x0, y0)
            sx1, sy1 = self._full_to_disp(x1, y1)
            cv2.rectangle(canvas, (self.split_x + sx0, sy0), (self.split_x + sx1, sy1), (0, 255, 0), 2)

    def show(self, preview=None):
        canvas = self._compose_canvas()
        # existing rects
        self.draw_rects_on_canvas(canvas, self.rects_full)
        # preview rect (yellow)
        if preview is not None:
            (x0, y0), (x1, y1) = preview
            x0, y0, x1, y1 = clamp_rect(x0, y0, x1, y1, self.w, self.h)
            sx0, sy0 = self._full_to_disp(x0, y0)
            sx1, sy1 = self._full_to_disp(x1, y1)
            cv2.rectangle(canvas, (sx0, sy0), (sx1, sy1), (0, 200, 255), 2)
            cv2.rectangle(canvas, (self.split_x + sx0, sy0), (self.split_x + sx1, sy1), (0, 200, 255), 2)

        # info text
        cv2.putText(canvas, f"{self.base}  |  rects: {len(self.rects_full)}",
                    (10, self.panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,50,255), 2, cv2.LINE_AA)
        cv2.imshow(self.window, canvas)


def main():
    basenames = list_common_basenames(IMG_DIR, MSK_DIR, IMG_EXT, MSK_EXT)
    if not basenames:
        print("No matching image/mask pairs found.")
        return

    idx = 0
    while 0 <= idx < len(basenames):
        base = basenames[idx]
        img_path = Path(IMG_DIR) / f"{base}{IMG_EXT}"
        msk_path = Path(MSK_DIR) / f"{base}{MSK_EXT}"

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            print(f"Skipping (failed to read): {base}")
            idx += 1
            continue

        if img.shape[:2] != msk.shape[:2]:
            print(f"Size mismatch for {base}: img {img.shape[:2]} vs msk {msk.shape[:2]} — resizing mask to image.")
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        ui = DualPanelCropper(img, msk, base)
        ui.show()

        while True:
            k = cv2.waitKeyEx(10)             # <-- use waitKeyEx, not waitKey
            if k == -1:
                continue

            if k & 0xFF == ord('u'):          # undo last
                if ui.rects_full:
                    ui.rects_full.pop()
                    ui.show()

            elif k & 0xFF == ord('c'):        # clear all
                ui.rects_full.clear()
                ui.show()

            elif k & 0xFF == ord('s'):        # save, stay
                if not ui.rects_full:
                    print("No rectangles to save.", flush=True)
                else:
                    saved = save_crops(base, ui.rects_full, ui.img, ui.msk,
                                    OUT_IMG_DIR, OUT_MSK_DIR, IMG_EXT, MSK_EXT)
                    print(f"Saved {saved} crop pair(s) for {base}.", flush=True)
                    ui.rects_full.clear()
                    ui.show()

            elif k & 0xFF == ord('n'):        # next without saving
                idx += 1
                break

            elif k & 0xFF == ord('p'):        # previous
                idx = max(0, idx - 1)
                break

            elif is_enter_key(k):             # Enter/Return: save (if any) + next
                if ui.rects_full:
                    saved = save_crops(base, ui.rects_full, ui.img, ui.msk,
                                    OUT_IMG_DIR, OUT_MSK_DIR, IMG_EXT, MSK_EXT)
                    print(f"Saved {saved} crop pair(s) for {base}.", flush=True)
                else:
                    print("No rectangles to save; moving to next.", flush=True)
                idx += 1
                break

            elif (k & 0xFF) == ord('q') or k == 27:  # quit
                cv2.destroyAllWindows()
                return

            # Optional: press '?' to print raw keycode (handy for debugging)
            elif (k & 0xFF) == ord('?'):
                print(f"Key code: {k}", flush=True)

    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    import numpy as np
    main()
