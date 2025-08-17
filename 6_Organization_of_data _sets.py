#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multispectral YOLO Dataset Builder (GUI)
- wybór źródła i celu
- skan + walidacja serii (RGB _D, G _MS_G, R _MS_R, RE _MS_RE, NIR _MS_NIR + .txt)
- dodatkowo: walidacja .txt -> brak danych (pusty plik) lub błędny format YOLO
- zatrzymanie przy: niekompletnych seriach (z informacją o folderze i NUMERACJĄ) lub duplikatach istotnych plików
- podział train/valid/test (procenty na poziomie serii)
- kopiowanie do RGB/G/R/RE/NIR/{images,labels}/{train,valid,test}
- raport CSV: base_name,split
"""

import os
import re
import csv
import shutil
import random
from pathlib import Path
from collections import defaultdict

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "Multispectral YOLO Dataset Builder"
RANDOM_SEED = 42

# --- Kanały: kanał -> (suffix, dopuszczalne rozszerzenia) ---
CHANNELS = {
    "RGB": ("_D", {".JPG", ".JPEG", ".PNG"}),
    "G":   ("_MS_G", {".TIF", ".TIFF"}),
    "R":   ("_MS_R", {".TIF", ".TIFF"}),
    "RE":  ("_MS_RE", {".TIF", ".TIFF"}),
    "NIR": ("_MS_NIR", {".TIF", ".TIFF"}),
}
CHAN_ORDER = ["RGB", "G", "R", "RE", "NIR"]

# --- Ignorowane śmieci/systemowe ---
IGNORED_FILENAMES = {"desktop.ini", "thumbs.db", ".ds_store"}
IGNORED_DIRNAME_TOKENS = {"__MACOSX"}

def norm_ext(p: Path) -> str:
    return p.suffix.upper()

def has_allowed_ext(p: Path, allowed: set) -> bool:
    return norm_ext(p) in allowed

def is_hidden_or_ignored(fname: str, dirpath: str) -> bool:
    if fname.startswith("."):
        return True
    if fname.lower() in IGNORED_FILENAMES:
        return True
    if any(tok in dirpath.upper() for tok in IGNORED_DIRNAME_TOKENS):
        return True
    return False

# --- Walidacja etykiet YOLO ---
def validate_label_file(path: Path):
    """
    Zwraca (ok: bool, error_msg: str | None).
    Kryteria:
      - plik nie może być pusty (po strip)
      - każda linia >= 5 kolumn: class x y w h
      - class = int >= 0; x,y,w,h ∈ [0,1]
    """
    try:
        txt = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        # spróbuj latin-1 awaryjnie
        txt = path.read_text(encoding="latin-1").strip()
    except Exception as e:
        return False, f"label read error: {e}"

    if txt == "":
        return False, "label file empty"

    for i, raw in enumerate(txt.splitlines(), start=1):
        line = raw.strip()
        if line == "":
            # pusta linia – ignorujemy, ale niech nie będzie jedyna
            continue
        parts = line.split()
        if len(parts) < 5:
            return False, f"line {i}: expected >=5 columns, got {len(parts)}"
        # class id
        try:
            cls = int(float(parts[0]))
            if cls < 0:
                return False, f"line {i}: negative class id"
        except Exception:
            return False, f"line {i}: class id not an integer"

        # bbox values in [0,1]
        try:
            x, y, w, h = map(float, parts[1:5])
        except Exception:
            return False, f"line {i}: non-numeric bbox"
        for val, name in zip([x, y, w, h], ["x", "y", "w", "h"]):
            if not (0.0 <= val <= 1.0):
                return False, f"line {i}: {name} out of [0,1] range"

    return True, None

def scan_source(root: Path):
    """
    Rekurencyjnie skanuje źródło.
    Zwraca:
      series_ok: dict base -> {"images": {ch: Path}, "labels": {ch: Path}}
      incomplete: dict base -> {"folder": str, "problems": [str,...]}
      duplicates: dict filename -> [fullpaths...]
    """
    chan_re = re.compile(r"(.*)_(D|MS_G|MS_R|MS_RE|MS_NIR)$", re.IGNORECASE)

    temp = defaultdict(lambda: {"images": defaultdict(list), "labels": defaultdict(list)})
    base_any_path = {}  # base -> dowolna ścieżka (do ustalenia folderu)
    file_locations = defaultdict(set)  # filename -> {fullpaths...}

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if is_hidden_or_ignored(fname, dirpath):
                continue

            p = Path(dirpath) / fname
            stem = p.stem
            ext_low = p.suffix.lower()

            # istotne etykiety .txt (do duplikatów)
            if ext_low == ".txt" and chan_re.match(stem):
                file_locations[fname].add(str(p))

            m = chan_re.match(stem)
            if not m:
                continue

            base_no_chan = m.group(1)
            suffix = "_" + m.group(2).upper()

            # obraz istotny?
            chan_key = None
            for ch, (suf, allowed_ext) in CHANNELS.items():
                if suffix == suf and has_allowed_ext(p, allowed_ext):
                    chan_key = ch
                    break
            if not chan_key:
                continue

            file_locations[fname].add(str(p))
            base_any_path.setdefault(base_no_chan, str(p))

            label_path = p.with_suffix(".txt")
            temp[base_no_chan]["images"][chan_key].append(p)
            if label_path.exists() and label_path.is_file():
                temp[base_no_chan]["labels"][chan_key].append(label_path)
                file_locations[label_path.name].add(str(label_path))

    series_ok = {}
    incomplete = {}

    for base, packs in temp.items():
        ok = True
        images = {}
        labels = {}
        problems = []

        # ustal folder serii (pierwszy znaleziony obraz/label)
        base_dir = None
        for ch in CHAN_ORDER:
            if packs["images"].get(ch):
                base_dir = str(packs["images"][ch][0].parent)
                break
        if base_dir is None:
            for ch in CHAN_ORDER:
                if packs["labels"].get(ch):
                    base_dir = str(packs["labels"][ch][0].parent)
                    break
        if base_dir is None:
            base_dir = str(Path(base_any_path.get(base, root)).parent)

        for ch in CHAN_ORDER:
            imgs = packs["images"].get(ch, [])
            labs = packs["labels"].get(ch, [])

            # obrazy
            if len(imgs) != 1:
                ok = False
                problems.append(f"channel {ch}: expected 1 image, found {len(imgs)}")
            else:
                images[ch] = imgs[0]

            # etykiety: obecność + walidacja zawartości
            if len(labs) != 1:
                ok = False
                problems.append(f"channel {ch}: expected 1 label, found {len(labs)}")
            else:
                lbl_path = labs[0]
                valid, err = validate_label_file(lbl_path)
                if not valid:
                    ok = False
                    problems.append(f"channel {ch}: {err} [{lbl_path.name}]")
                labels[ch] = lbl_path

        if ok:
            series_ok[base] = {"images": images, "labels": labels}
        else:
            incomplete[base] = {"folder": base_dir, "problems": problems}

    duplicates = {fn: sorted(paths) for fn, paths in file_locations.items() if len(paths) > 1}
    return series_ok, incomplete, duplicates

def ensure_structure(dst_root: Path):
    for ch in CHAN_ORDER:
        for kind in ["images", "labels"]:
            for split in ["train", "valid", "test"]:
                (dst_root / ch / kind / split).mkdir(parents=True, exist_ok=True)

def compute_split_counts(n_series: int, pct_train: float, pct_valid: float, pct_test: float):
    if abs((pct_train + pct_valid + pct_test) - 100.0) > 1e-6:
        raise ValueError("Suma procentów musi wynosić 100.")
    t = int(n_series * pct_train / 100.0)
    v = int(n_series * pct_valid / 100.0)
    s = n_series - t - v
    return {"train": t, "valid": v, "test": s}

def pick_and_assign(series_bases, split_counts):
    random.seed(RANDOM_SEED)
    shuffled = series_bases[:]
    random.shuffle(shuffled)
    assign = {}
    i = 0
    for split in ["train", "valid", "test"]:
        k = split_counts[split]
        for _ in range(k):
            if i >= len(shuffled):
                break
            assign[shuffled[i]] = split
            i += 1
    return assign

def copy_series(assign_map, series, dst_root: Path):
    ensure_structure(dst_root)
    for base, split in assign_map.items():
        pack = series[base]
        for ch in CHAN_ORDER:
            img_src = pack["images"][ch]
            lbl_src = pack["labels"][ch]
            img_dst = dst_root / ch / "images" / split / img_src.name
            lbl_dst = dst_root / ch / "labels" / split / (img_src.with_suffix(".txt").name)
            if img_dst.exists() or lbl_dst.exists():
                raise FileExistsError(f"Plik docelowy już istnieje: {img_dst} lub {lbl_dst}")
            shutil.copy2(img_src, img_dst)
            shutil.copy2(lbl_src, lbl_dst)

def write_report_csv(assign_map, dst_root: Path):
    report_path = dst_root / "report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["base_name", "split"])
        for base, split in sorted(assign_map.items()):
            w.writerow([base, split])
    return report_path

# --------- Scrollowalne okno ---------

def show_scrollable_message(title: str, content: str):
    win = tk.Toplevel()
    win.title(title)
    win.geometry("900x500")
    win.transient()
    win.grab_set()

    frm = ttk.Frame(win)
    frm.pack(fill="both", expand=True)

    yscroll = ttk.Scrollbar(frm, orient="vertical")
    txt = tk.Text(frm, wrap="none", yscrollcommand=yscroll.set)
    yscroll.config(command=txt.yview)
    yscroll.pack(side="right", fill="y")
    txt.pack(side="left", fill="both", expand=True)

    txt.insert("1.0", content)
    txt.config(state="disabled")

    ttk.Button(win, text="OK", command=win.destroy).pack(pady=8)
    win.wait_window()

# -------------------- GUI --------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("860x600")

        self.src_path = tk.StringVar()
        self.dst_path = tk.StringVar()

        self.n_series = tk.IntVar(value=0)
        self.n_images = tk.IntVar(value=0)
        self.n_labels = tk.IntVar(value=0)

        self.pct_train = tk.DoubleVar(value=70.0)
        self.pct_valid = tk.DoubleVar(value=20.0)
        self.pct_test  = tk.DoubleVar(value=10.0)

        self.series = {}
        self.assign_map = {}

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        src_frame = ttk.LabelFrame(self, text="Katalog źródłowy (rekurencyjnie)")
        src_frame.pack(fill="x", **pad)
        ttk.Entry(src_frame, textvariable=self.src_path).pack(side="left", fill="x", expand=True, padx=6, pady=6)
        ttk.Button(src_frame, text="Wybierz…", command=self.choose_src).pack(side="right", padx=6, pady=6)

        dst_frame = ttk.LabelFrame(self, text="Katalog docelowy (struktura RGB/G/R/RE/NIR)")
        dst_frame.pack(fill="x", **pad)
        ttk.Entry(dst_frame, textvariable=self.dst_path).pack(side="left", fill="x", expand=True, padx=6, pady=6)
        ttk.Button(dst_frame, text="Wybierz…", command=self.choose_dst).pack(side="right", padx=6, pady=6)

        scan_frame = ttk.Frame(self)
        scan_frame.pack(fill="x", **pad)
        ttk.Button(scan_frame, text="Skanuj źródło", command=self.scan_action).pack(side="left")
        self.summary_lbl = ttk.Label(scan_frame, text="Brak danych. Najpierw zeskanuj źródło.")
        self.summary_lbl.pack(side="left", padx=12)

        pct_frame = ttk.LabelFrame(self, text="Podział procentowy (na poziomie serii / pojedynczego kanału)")
        pct_frame.pack(fill="x", **pad)
        ttk.Label(pct_frame, text="train [%]:").grid(row=0, column=0, sticky="e")
        ttk.Entry(pct_frame, width=6, textvariable=self.pct_train).grid(row=0, column=1, sticky="w")
        ttk.Label(pct_frame, text="valid [%]:").grid(row=0, column=2, sticky="e")
        ttk.Entry(pct_frame, width=6, textvariable=self.pct_valid).grid(row=0, column=3, sticky="w")
        ttk.Label(pct_frame, text="test [%]:").grid(row=0, column=4, sticky="e")
        ttk.Entry(pct_frame, width=6, textvariable=self.pct_test).grid(row=0, column=5, sticky="w")
        ttk.Button(pct_frame, text="Przelicz podział", command=self.compute_splits).grid(row=0, column=6, padx=10)

        table_frame = ttk.LabelFrame(self, text="Propozycja podziału")
        table_frame.pack(fill="both", expand=False, **pad)
        self.tree = ttk.Treeview(table_frame, columns=("split", "count"), show="headings", height=4)
        self.tree.heading("split", text="Split")
        self.tree.heading("count", text="Liczba serii")
        self.tree.column("split", width=120, anchor="center")
        self.tree.column("count", width=140, anchor="e")
        self.tree.pack(fill="x", padx=6, pady=6)

        actions = ttk.Frame(self)
        actions.pack(fill="x", **pad)
        ttk.Button(actions, text="Kopiuj do celu", command=self.copy_action).pack(side="left")
        ttk.Button(actions, text="Zakończ", command=self.destroy).pack(side="right")

        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(log_frame, height=10)
        self.log.pack(fill="both", expand=True, padx=6, pady=6)

    def choose_src(self):
        path = filedialog.askdirectory(title="Wybierz katalog źródłowy")
        if path:
            self.src_path.set(path)

    def choose_dst(self):
        path = filedialog.askdirectory(title="Wybierz katalog docelowy")
        if path:
            self.dst_path.set(path)

    def log_write(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.update_idletasks()

    def scan_action(self):
        src = self.src_path.get().strip()
        if not src:
            messagebox.showerror(APP_TITLE, "Wskaż katalog źródłowy.")
            return
        root = Path(src)
        if not root.exists():
            messagebox.showerror(APP_TITLE, "Katalog źródłowy nie istnieje.")
            return

        self.log_write(f"[SKAN] Start: {root}")
        series_ok, incomplete, duplicates = scan_source(root)

        if duplicates:
            details = []
            for fname, paths in sorted(duplicates.items()):
                details.append(fname)
                for p in paths:
                    details.append(f"    {p}")
            show_scrollable_message(APP_TITLE, "Wykryto zduplikowane nazwy plików (muszą być unikalne):\n\n" + "\n".join(details))
            self.log_write("[BŁĄD] Zduplikowane pliki:\n" + "\n".join(details))
            return

        if incomplete:
            lines = []
            lines.append(f"Liczba niekompletnych serii: {len(incomplete)}\n")
            for idx, (base, info) in enumerate(sorted(incomplete.items()), start=1):
                lines.append(f"{idx}) {base}")
                lines.append(f"    folder: {info['folder']}")
                for pr in info["problems"]:
                    lines.append(f"    - {pr}")
                lines.append("")  # odstęp
            text = "Wykryto niekompletne serie. Najpierw popraw dane.\n\n" + "\n".join(lines)
            show_scrollable_message(APP_TITLE, text)
            self.log_write(f"[OSTRZEŻENIE] Niekompletne serie: {len(incomplete)}")
            return

        self.series = series_ok
        n_series = len(series_ok)
        self.n_series.set(n_series)
        self.n_images.set(n_series * len(CHAN_ORDER))
        self.n_labels.set(n_series * len(CHAN_ORDER))

        self.summary_lbl.config(text=f"Serie: {n_series} | Obrazy (łącznie): {self.n_images.get()} | Etykiety (łącznie): {self.n_labels.get()}")
        self.log_write(f"[OK] Znaleziono kompletnych serii: {n_series}")
        self.compute_splits()

    def compute_splits(self):
        n = self.n_series.get()
        if n == 0:
            return
        try:
            counts = compute_split_counts(n, self.pct_train.get(), self.pct_valid.get(), self.pct_test.get())
        except Exception as e:
            messagebox.showerror(APP_TITLE, str(e))
            return

        for row in self.tree.get_children():
            self.tree.delete(row)
        for split in ["train", "valid", "test"]:
            self.tree.insert("", "end", values=(split, counts[split]))

        bases = list(self.series.keys())
        self.assign_map = pick_and_assign(bases, counts)
        self.log_write("[INFO] Zaktualizowano podział (podgląd).")

    def copy_action(self):
        if not self.series:
            messagebox.showinfo(APP_TITLE, "Brak zeskanowanych danych.")
            return
        dst = self.dst_path.get().strip()
        if not dst:
            messagebox.showerror(APP_TITLE, "Wskaż katalog docelowy.")
            return
        dst_root = Path(dst)
        dst_root.mkdir(parents=True, exist_ok=True)

        try:
            counts = compute_split_counts(self.n_series.get(), self.pct_train.get(), self.pct_valid.get(), self.pct_test.get())
        except Exception as e:
            messagebox.showerror(APP_TITLE, str(e))
            return
        bases = list(self.series.keys())
        assign_map = pick_and_assign(bases, counts)

        try:
            copy_series(assign_map, self.series, dst_root)
        except FileExistsError as e:
            show_scrollable_message(APP_TITLE, f"Kolizja w katalogu docelowym:\n\n{e}")
            self.log_write(f"[BŁĄD] {e}")
            return
        except Exception as e:
            show_scrollable_message(APP_TITLE, f"Błąd kopiowania:\n\n{e}")
            self.log_write(f"[BŁĄD] {e}")
            return

        report_path = write_report_csv(assign_map, dst_root)
        messagebox.showinfo(APP_TITLE, f"Zakończono kopiowanie.\nRaport: {report_path}")
        self.log_write(f"[OK] Zakończono. Raport: {report_path}")

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
