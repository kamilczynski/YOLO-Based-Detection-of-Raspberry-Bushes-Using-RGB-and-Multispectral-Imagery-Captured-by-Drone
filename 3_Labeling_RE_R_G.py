#!/usr/bin/env python3
"""
propagate_yolo_homography.py  – rev-6 (auto-choose matrix order)
================================================================
*Problem:* w niektórych ujęciach CalibratedHMatrix wygląda na to, że DJI
istnieją dwie konwencje („rig→sensor” vs „sensor→rig”).  Zamiast zgadywać,
**policzymy obie transformacje** dla każdego pasma i wybierzemy tę, której
szybka korelacja szablonu (`cv2.matchTemplate`) daje lepszy wynik.

Algorytm:
1. Dla pasma docelowego wyliczamy dwie macierze kandydatki:
   * `T1 =  H_tar  · H_ref⁻¹`
   * `T2 =  H_ref⁻¹ · H_tar`
2. Z każdej przewidywanej ramki wycinamy okno (±20 px) i liczymy wartość
   maksymalnej korelacji (`score`).
3. Wybieramy T* z większym `score`; następnie wyliczamy Δx/Δy i aktualizujemy
   `xc, yc`.

Dzięki temu nie musimy wiedzieć, którą orientację macierzy stosuje DJI – kod
sam dopasuje tę, która faktycznie „siada” na obrazku.

*Wymagania*: `opencv-python` ≥ 4 (do template matching).  Bez niego skrypt
zachowuje się jak **rev-4** (używa T1).
"""

import argparse
import glob
import json
import os
import subprocess
import warnings
from collections import defaultdict

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # will run in fallback mode

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None

from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# native sensor resolution
orig_w, orig_h = 2592, 1944

# ---------------------------------------------------------------------------
# EXIF utilities
# ---------------------------------------------------------------------------

def exif_json(path: str) -> dict:
    cmd = [
        "exiftool",
        "-CalibratedHMatrix#",
        "-ImageWidth",
        "-ImageHeight",
        "-n",
        "-json",
        path,
    ]
    return json.loads(subprocess.check_output(cmd, text=True))[0]


def to_mat(raw) -> np.ndarray:
    vals = [float(v) for v in (raw.split(",") if isinstance(raw, str) else raw)]
    return np.array(vals, dtype=np.float64).reshape(3, 3)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def band_data(info):
    H = to_mat(info["CalibratedHMatrix"])
    w_img = int(info["ImageWidth"])
    h_img = int(info["ImageHeight"])
    S = np.diag([orig_w / w_img, orig_h / h_img, 1.0])  # sensor = S · img
    return H, S, w_img, h_img


def clamp01(v):
    return max(0.0, min(1.0, v))


def group_tifs(folder, bands):
    groups = defaultdict(dict)
    for tif in glob.glob(os.path.join(folder, "*_MS_*.TIF")):
        name = os.path.basename(tif)
        if "_MS_" not in name:  # safety
            continue
        prefix, rest = name.split("_MS_", 1)
        band = rest.split(".", 1)[0]
        if band in bands:
            groups[prefix][band] = tif
    return groups

# ---------------------------------------------------------------------------
# template-matching utilities (needs OpenCV)
# ---------------------------------------------------------------------------

def match_score(search, template):
    """Return (score, dx, dy) – where (dx,dy) shift template center to best pos."""
    if search.size == 0 or template.size == 0:
        return -1.0, 0.0, 0.0
    res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    dx = (max_loc[0] + template.shape[1] / 2) - (search.shape[1] / 2)
    dy = (max_loc[1] + template.shape[0] / 2) - (search.shape[0] / 2)
    return max_val, dx, dy

# ---------------------------------------------------------------------------
# main propagation
# ---------------------------------------------------------------------------

def propagate(folder: str, bands, ref_band: str):
    groups = group_tifs(folder, bands)
    if not groups:
        print("[WARN] No *_MS_*.TIF found")
        return

    for prefix, paths in groups.items():
        if ref_band not in paths:
            continue
        labels_ref = os.path.join(folder, f"{prefix}_MS_{ref_band}.txt")
        if not os.path.exists(labels_ref):
            continue
        with open(labels_ref, "r", encoding="utf-8") as fh:
            ref_labels = [ln.strip().split() for ln in fh if ln.strip()]
        if not ref_labels:
            continue

        info_r = exif_json(paths[ref_band])
        H_r, S_r, w_r, h_r = band_data(info_r)
        H_r_inv = np.linalg.inv(H_r)

        # load reference image if OpenCV available
        ref_img = cv2.imread(paths[ref_band], cv2.IMREAD_GRAYSCALE) if cv2 else None

        for band in bands:
            if band == ref_band or band not in paths:
                continue
            info_t = exif_json(paths[band])
            H_t, S_t, w_t, h_t = band_data(info_t)

            # two candidate transforms
            T1 = np.linalg.inv(S_t) @ (H_t @ H_r_inv) @ S_r
            T2 = np.linalg.inv(S_t) @ (H_r_inv @ H_t) @ S_r

            tar_img = cv2.imread(paths[band], cv2.IMREAD_GRAYSCALE) if cv2 else None

            out_path = os.path.join(folder, f"{prefix}_MS_{band}.txt")
            with open(out_path, "w", encoding="utf-8") as wf:
                for cls, xc, yc, bw, bh in ref_labels:
                    xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
                    # reference bbox in pixel coords
                    x0r = int(round((xc - bw / 2) * w_r))
                    y0r = int(round((yc - bh / 2) * h_r))
                    x1r = int(round((xc + bw / 2) * w_r))
                    y1r = int(round((yc + bh / 2) * h_r))

                    corners = np.array([[x0r, y0r, 1], [x1r, y0r, 1], [x1r, y1r, 1], [x0r, y1r, 1]]).T

                    best_dx = best_dy = 0.0
                    if cv2 is not None:
                        # predict with both transforms
                        picks = []
                        for T in (T1, T2):
                            pts_t = T @ corners
                            pts_t = pts_t[:2] / pts_t[2]
                            x_min, y_min = pts_t.min(axis=1)
                            x_max, y_max = pts_t.max(axis=1)
                            x0t = int(round(x_min)); y0t = int(round(y_min))
                            x1t = int(round(x_max)); y1t = int(round(y_max))
                            # build search window
                            margin = 20
                            sx0 = max(0, x0t - margin); sy0 = max(0, y0t - margin)
                            sx1 = min(w_t, x1t + margin); sy1 = min(h_t, y1t + margin)
                            template = ref_img[y0r:y1r, x0r:x1r]
                            search = tar_img[sy0:sy1, sx0:sx1]
                            score, dx, dy = match_score(search, template)
                            picks.append((score, dx, dy, x0t, y0t, x1t, y1t))
                        # choose better
                        picks.sort(key=lambda v: v[0], reverse=True)
                        score, dx, dy, x0t, y0t, x1t, y1t = picks[0]
                        x0t += dx; x1t += dx; y0t += dy; y1t += dy
                    else:
                        # OpenCV unavailable – fallback to T1
                        pts_t = T1 @ corners
                        pts_t = pts_t[:2] / pts_t[2]
                        x_min, y_min = pts_t.min(axis=1)
                        x_max, y_max = pts_t.max(axis=1)
                        x0t, y0t, x1t, y1t = x_min, y_min, x_max, y_max

                    # to YOLO
                    xc_t = (x0t + x1t) / 2 / w_t
                    yc_t = (y0t + y1t) / 2 / h_t
                    bw_t = (x1t - x0t) / w_t
                    bh_t = (y1t - y0t) / h_t

                    wf.write(f"{cls} {clamp01(xc_t):.6f} {clamp01(yc_t):.6f} {bw_t:.6f} {bh_t:.6f}\n")
            print("[OK]", out_path)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Propagate YOLO labels (auto matrix order, optional OpenCV)")
    ap.add_argument("-d", "--data_folder")
    ap.add_argument("-r", "--reference_band", default="NIR", choices=["NIR", "G", "R", "RE"])
    args = ap.parse_args()

    if not args.data_folder:
        if tk is None:
            raise SystemExit("--data_folder required (no tkinter)")
        root = tk.Tk(); root.withdraw()
        args.data_folder = filedialog.askdirectory(title="Choose data folder")
        if not args.data_folder:
            raise SystemExit("No folder selected – aborting")

    propagate(args.data_folder, ["NIR", "G", "R", "RE"], args.reference_band)


if __name__ == "__main__":
    main()
