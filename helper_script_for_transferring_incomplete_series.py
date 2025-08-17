#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Move Incomplete Multispectral Series (Simple GUI)
- Wybierz źródło i folder docelowy
- Skanuje rekurencyjnie serie DJI_*_{_D,_MS_G,_MS_R,_MS_RE,_MS_NIR}
- Jeśli seria jest niekompletna (brakuje obrazu lub jego .txt), PRZENOSI (shutil.move) wszystkie pliki tej serii do celu
- W źródle pozostają tylko serie kompletne
- Zachowuje strukturę podfolderów względem źródła
"""

import os
import re
import shutil
from pathlib import Path
from collections import defaultdict

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "Move Incomplete Multispectral Series"

# Definicja kanałów: nazwa -> (suffix, dopuszczalne rozszerzenia)
CHANNELS = {
    "RGB": ("_D", {".JPG", ".JPEG", ".PNG"}),
    "G":   ("_MS_G", {".TIF", ".TIFF"}),
    "R":   ("_MS_R", {".TIF", ".TIFF"}),
    "RE":  ("_MS_RE", {".TIF", ".TIFF"}),
    "NIR": ("_MS_NIR", {".TIF", ".TIFF"}),
}
CHAN_ORDER = ["RGB", "G", "R", "RE", "NIR"]

IGNORED_FILENAMES = {"desktop.ini", "thumbs.db", ".ds_store"}
IGNORED_DIR_TOKENS = {"__MACOSX"}

def is_ignored(dirpath: str, fname: str) -> bool:
    if fname.startswith("."):
        return True
    if fname.lower() in IGNORED_FILENAMES:
        return True
    if any(tok in dirpath.upper() for tok in IGNORED_DIR_TOKENS):
        return True
    return False

def scan_series(source_root: Path):
    """
    Zwraca:
      series: base_name -> {"images": {ch: Path}, "labels": {ch: Path}}
      incomplete: set(base_name)
    """
    chan_re = re.compile(r"(.*)_(D|MS_G|MS_R|MS_RE|MS_NIR)$", re.IGNORECASE)
    temp = defaultdict(lambda: {"images": defaultdict(list), "labels": defaultdict(list)})

    for dirpath, _, filenames in os.walk(source_root):
        for fname in filenames:
            if is_ignored(dirpath, fname):
                continue
            p = Path(dirpath) / fname
            stem = p.stem
            ext = p.suffix.upper()

            m = chan_re.match(stem)
            if not m:
                continue

            base = m.group(1)
            suffix = "_" + m.group(2).upper()

            # dopasuj kanał i akceptowane rozszerzenie
            ch_key = None
            for ch, (suf, allowed) in CHANNELS.items():
                if suffix == suf and ext in allowed:
                    ch_key = ch
                    break
            if not ch_key:
                continue

            # dodaj obraz
            temp[base]["images"][ch_key].append(p)
            # dodaj label jeśli istnieje
            lbl = p.with_suffix(".txt")
            if lbl.exists() and lbl.is_file():
                temp[base]["labels"][ch_key].append(lbl)

    series = {}
    incomplete = set()

    for base, packs in temp.items():
        ok = True
        images = {}
        labels = {}
        for ch in CHAN_ORDER:
            imgs = packs["images"].get(ch, [])
            labs = packs["labels"].get(ch, [])
            if len(imgs) != 1 or len(labs) != 1:
                ok = False
            if imgs:
                images[ch] = imgs[0]
            if labs:
                labels[ch] = labs[0]
        if ok:
            series[base] = {"images": images, "labels": labels}
        else:
            incomplete.add(base)

    return series, incomplete, temp  # temp zawiera też ścieżki do plików, które trzeba będzie przenieść

def move_incomplete(source_root: Path, dest_root: Path, incomplete_bases, temp_map):
    """
    Przenosi WSZYSTKIE pliki należące do niekompletnej serii:
    - obrazy 5 kanałów (jeśli są)
    - odpowiadające im .txt (jeśli są)
    Zachowuje strukturę podfolderów względem source_root.
    Zwraca liczbę przeniesionych plików.
    """
    moved_files = 0
    for base in incomplete_bases:
        packs = temp_map[base]
        # zbierz potencjalne ścieżki: obrazy i znalezione etykiety
        paths = []
        for ch in CHAN_ORDER:
            for p in packs["images"].get(ch, []):
                paths.append(p)
                lbl = p.with_suffix(".txt")
                if lbl.exists():
                    paths.append(lbl)
        # usuń duplikaty
        uniq_paths = sorted(set(paths), key=lambda x: str(x).lower())

        for src in uniq_paths:
            # ścieżka względna od źródła
            rel = src.relative_to(source_root)
            dst = dest_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.move(str(src), str(dst))
                moved_files += 1
            else:
                # jeśli plik już istnieje w docelowym, nadaj sufiks, aby nie blokować procesu
                dst_alt = dst.with_name(dst.stem + "_moved" + dst.suffix)
                shutil.move(str(src), str(dst_alt))
                moved_files += 1
    return moved_files

# ---------------- GUI ----------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("780x420")

        self.src = tk.StringVar()
        self.dst = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        f1 = ttk.LabelFrame(self, text="Ścieżki")
        f1.pack(fill="x", **pad)
        ttk.Label(f1, text="Źródło:").grid(row=0, column=0, sticky="e")
        ttk.Entry(f1, textvariable=self.src, width=80).grid(row=0, column=1, sticky="we")
        ttk.Button(f1, text="Wybierz…", command=self.choose_src).grid(row=0, column=2, padx=6)

        ttk.Label(f1, text="Cel (tu przenieść niepełne):").grid(row=1, column=0, sticky="e")
        ttk.Entry(f1, textvariable=self.dst, width=80).grid(row=1, column=1, sticky="we")
        ttk.Button(f1, text="Wybierz…", command=self.choose_dst).grid(row=1, column=2, padx=6)

        f2 = ttk.LabelFrame(self, text="Akcje")
        f2.pack(fill="x", **pad)
        ttk.Button(f2, text="Przenieś niepełne serie", command=self.run).pack(side="left")

        self.info = ttk.Label(self, text="Gotowy.")
        self.info.pack(fill="x", **pad)

        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(log_frame, height=10)
        self.log.pack(fill="both", expand=True, padx=6, pady=6)

    def choose_src(self):
        p = filedialog.askdirectory(title="Wybierz katalog źródłowy")
        if p: self.src.set(p)

    def choose_dst(self):
        p = filedialog.askdirectory(title="Wybierz katalog docelowy")
        if p: self.dst.set(p)

    def log_write(self, s: str):
        self.log.insert("end", s + "\n")
        self.log.see("end")
        self.update_idletasks()

    def run(self):
        src = self.src.get().strip()
        dst = self.dst.get().strip()
        if not src or not dst:
            messagebox.showerror(APP_TITLE, "Wskaż katalog źródłowy i docelowy.")
            return
        srcp = Path(src)
        dstp = Path(dst)
        if not srcp.exists():
            messagebox.showerror(APP_TITLE, "Katalog źródłowy nie istnieje.")
            return
        dstp.mkdir(parents=True, exist_ok=True)

        self.info.config(text="Skanowanie…")
        self.log_write(f"[INFO] Skanuję: {srcp}")
        series_ok, incomplete, temp_map = scan_series(srcp)

        total_series = len(series_ok) + len(incomplete)
        self.log_write(f"[INFO] Serie łącznie: {total_series}")
        self.log_write(f"[INFO] Pełne: {len(series_ok)} | Niepełne: {len(incomplete)}")

        if not incomplete:
            messagebox.showinfo(APP_TITLE, "Brak niepełnych serii. Nic do przeniesienia.")
            self.info.config(text="Brak niepełnych serii.")
            return

        # krótki podgląd pierwszych kilkunastu do logu
        for i, base in enumerate(sorted(incomplete)):
            if i >= 20:
                self.log_write("… (pominięto resztę w logu)")
                break
            self.log_write(f" - {base}")

        self.info.config(text="Przenoszenie…")
        moved = move_incomplete(srcp, dstp, incomplete, temp_map)

        messagebox.showinfo(APP_TITLE, f"Przeniesiono plików: {moved}\nNiepełnych serii: {len(incomplete)}")
        self.info.config(text=f"Gotowe. Przeniesiono plików: {moved} | Niepełnych serii: {len(incomplete)}")
        self.log_write(f"[OK] Zakończono. Przeniesiono plików: {moved}")

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
