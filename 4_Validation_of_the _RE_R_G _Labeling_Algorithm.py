#!/usr/bin/env python3
"""
validate_yolo_gui.py – rev-2
===========================
Walidacja dwóch zestawów etykiet YOLO (GT vs PR) z GUI do wyboru folderów.
Dodane:
* **Wielokrotne progi IoU** – domyślnie 0.5, 0.75, 0.9 (można podać listę).
* **Średni błąd Δx, Δy w pikselach** i ich odchylenie standardowe.

Przykłady:
-----------
CLI bez GUI:
```
python validate_yolo_gui.py manual/ auto/ --iou 0.5 0.75 0.9
```
Jeśli nie podasz ścieżek – skrypt poprosi o nie w oknach.
"""
import argparse, glob, os, sys, statistics, numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except ImportError:
    tk = None

# ---------------------------------------------------------------------------
# YOLO helpers
# ---------------------------------------------------------------------------

def read_yolo(path):
    arr = []
    if not os.path.exists(path):
        return np.zeros((0, 5))
    with open(path) as fh:
        for ln in fh:
            if ln.strip():
                arr.append([float(x) for x in ln.split()])
    return np.asarray(arr, dtype=float)


def iou_xywh(a, b):
    xa0, ya0 = a[0] - a[2] / 2, a[1] - a[3] / 2
    xa1, ya1 = a[0] + a[2] / 2, a[1] + a[3] / 2
    xb0, yb0 = b[0] - b[2] / 2, b[1] - b[3] / 2
    xb1, yb1 = b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0, min(xa1, xb1) - max(xa0, xb0)) * max(0, min(ya1, yb1) - max(ya0, yb0))
    if inter == 0:
        return 0.0
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


# ---------------------------------------------------------------------------
# Matching & metrics
# ---------------------------------------------------------------------------

def match_single(gt, pr):
    """Greedy 1-to-1 matching ignoring class if arrays small.
    Returns list of (IoU, dx, dy) for each hit, FN, FP"""
    used = set(); hits = []
    for g in gt:
        best_iou, best_j = -1, -1
        for j, p in enumerate(pr):
            if j in used or int(p[0]) != int(g[0]):  # class must match
                continue
            iou = iou_xywh(g[1:], p[1:])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= 0:
            used.add(best_j)
            # dx,dy w pikselach sensora (przyjmując 0-1 * 2592/1944)
            dx = (p[1] - g[1]) * 2592
            dy = (p[2] - g[2]) * 1944
            hits.append((best_iou, dx, dy))
    fn = len(gt) - len(hits)
    fp = len(pr) - len(used)
    return hits, fn, fp

# ---------------------------------------------------------------------------
# GUI util
# ---------------------------------------------------------------------------

def choose_folder(prompt):
    if tk is None:
        print("[ERR] tkinter not available; pass folders in CLI")
        sys.exit(1)
    root = tk.Tk(); root.withdraw()
    path = filedialog.askdirectory(title=prompt)
    if not path:
        messagebox.showerror('validate_yolo', 'Folder not selected – aborting.')
        sys.exit(1)
    return path

# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate(dir_gt, dir_pr, iou_thrs):
    files = sorted(glob.glob(os.path.join(dir_gt, '*_MS_*.txt')))
    if not files:
        print('[ERR] No *_MS_*.txt in', dir_gt); sys.exit(1)

    data = {thr: {'hits':0,'fn':0,'fp':0} for thr in iou_thrs}
    dx_all, dy_all = [], []

    for f in files:
        name = os.path.basename(f)
        gt = read_yolo(f)
        pr = read_yolo(os.path.join(dir_pr, name))
        hits, fn, fp = match_single(gt, pr)
        # store dx/dy if any
        for _, dx, dy in hits:
            dx_all.append(dx); dy_all.append(dy)
        for thr in iou_thrs:
            good = [h for h in hits if h[0] >= thr]
            data[thr]['hits'] += len(good)
            data[thr]['fn']   += fn
            data[thr]['fp']   += fp

    print('Images:', len(files))
    if dx_all:
        print(f"Δx  mean±σ : {statistics.mean(map(abs,dx_all)):.2f} ± {statistics.stdev(map(abs,dx_all)):.2f} px")
        print(f"Δy  mean±σ : {statistics.mean(map(abs,dy_all)):.2f} ± {statistics.stdev(map(abs,dy_all)):.2f} px")
    for thr in sorted(iou_thrs):
        d = data[thr]
        rec = d['hits'] / (d['hits']+d['fn']) if d['hits']+d['fn'] else 0
        prec= d['hits'] / (d['hits']+d['fp']) if d['hits']+d['fp'] else 0
        print(f"IoU≥{thr:0.2f}  Recall: {rec:.3f}   Precision: {prec:.3f}   FN:{d['fn']} FP:{d['fp']}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Validate YOLO folders (GUI or CLI)')
    ap.add_argument('gt', nargs='?', help='GT folder')
    ap.add_argument('pr', nargs='?', help='PR folder')
    ap.add_argument('--iou', type=float, nargs='+', default=[0.5,0.75,0.9], help='IoU thresholds list')
    args = ap.parse_args()

    dir_gt = args.gt or choose_folder('Wybierz folder z etykietami GT (manual)')
    dir_pr = args.pr or choose_folder('Wybierz folder z etykietami PR (auto)')

    validate(dir_gt, dir_pr, args.iou)

if __name__ == '__main__':
    main()
