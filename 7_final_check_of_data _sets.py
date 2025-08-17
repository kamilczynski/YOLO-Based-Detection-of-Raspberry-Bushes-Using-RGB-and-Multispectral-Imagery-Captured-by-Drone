#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multispectral Dataset Validator (GUI) — cross-channel by base name (fixed)
- wybieramy katalogi dla RGB, G, R, RE, NIR (root każdego kanału zawiera images/ i labels/)
- walidacja wewnątrz kanału (images vs labels, poprawność YOLO .txt)
- walidacja między kanałami po bazie nazwy (ignorując sufiks kanału: _D, _MS_G, _MS_R, _MS_RE, _MS_NIR, …)
- raport w GUI + okna przewijane ze szczegółami
"""

import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "Multispectral Dataset Validator"
CHANNELS = ["RGB", "G", "R", "RE", "NIR"]
SPLITS = ["train", "valid", "test"]

# --- Lista rozpoznawanych sufiksów kanałów (najdłuższe -> najkrótsze) ---
SUFFIXES = [
    "_MS_NIR", "_MS_RE", "_MS_G", "_MS_R",
    "_NIR", "_RE", "_RGB", "_R", "_G", "_D"
]

def stem_to_base(stem: str):
    """Zwraca bazę nazwy bez sufiksu kanału; None, jeśli nie rozpoznano sufiksu."""
    up = stem.upper()
    for suf in SUFFIXES:
        if up.endswith(suf):
            return stem[: -len(suf)]
    return None

# --- Walidacja YOLO label ---
def validate_label_file(path: Path):
    """Sprawdź poprawność pliku YOLO .txt."""
    try:
        txt = path.read_text(encoding="utf-8").strip()
    except Exception as e:
        return False, f"cannot read ({e})"
    if txt == "":
        return False, "empty file"
    for i, raw in enumerate(txt.splitlines(), start=1):
        parts = raw.strip().split()
        if len(parts) < 5:
            return False, f"line {i}: <5 columns"
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
        except Exception:
            return False, f"line {i}: parse error"
        if cls < 0:
            return False, f"line {i}: negative class id"
        for val, name in zip([x, y, w, h], ["x", "y", "w", "h"]):
            if not (0.0 <= val <= 1.0):
                return False, f"line {i}: {name} out of [0,1]"
    return True, None

# --- Skan jednego kanału ---
def scan_dataset(root: Path):
    """
    Zwraca:
    {
      "series_stem": {split: set(stemów z kompletną parą img+label i poprawną etykietą)},
      "series_base": {split: set(baz bez sufiksu kanału dla powyższych stemów)},
      "errors": [string,...]
    }
    """
    results_stem = {s: set() for s in SPLITS}
    results_base = {s: set() for s in SPLITS}
    errors = []

    for split in SPLITS:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            errors.append(f"[{root.name}] split '{split}': missing 'images/' or 'labels/'")
            continue

        img_stems = {p.stem for p in img_dir.iterdir() if p.is_file()}
        lbl_paths = {p.stem: p for p in lbl_dir.glob("*.txt") if p.is_file()}
        lbl_stems = set(lbl_paths.keys())

        # brakujące pary
        for s in sorted(img_stems - lbl_stems):
            errors.append(f"[{root.name}] {split}: missing label for '{s}'")
        for s in sorted(lbl_stems - img_stems):
            errors.append(f"[{root.name}] {split}: missing image for '{s}'")

        # walidacja .txt i budowa zbiorów
        for s in sorted(img_stems & lbl_stems):
            ok, msg = validate_label_file(lbl_paths[s])
            if not ok:
                errors.append(f"[{root.name}] {split}: bad label '{s}.txt' -> {msg}")
                continue
            results_stem[split].add(s)

            base = stem_to_base(s)
            if base is None:
                errors.append(f"[{root.name}] {split}: unexpected filename (cannot derive base) '{s}'")
            else:
                results_base[split].add(base)

    return {"series_stem": results_stem, "series_base": results_base, "errors": errors}

# --- Porównanie między kanałami (po bazie nazwy) ---
def compare_across_channels(data_by_channel):
    """
    data_by_channel[ch] -> wynik scan_dataset
    Sprawdza spójność po 'series_base' w każdym splicie.
    """
    errors = []
    for split in SPLITS:
        base_sets = {ch: data["series_base"][split] for ch, data in data_by_channel.items()}

        union = set().union(*base_sets.values()) if base_sets else set()
        if not union:
            continue

        # punktowo: które bazy nie są obecne we wszystkich kanałach
        for base in sorted(union):
            present = [ch for ch, s in base_sets.items() if base in s]
            missing = [ch for ch, s in base_sets.items() if base not in s]
            if missing and present:
                errors.append(f"[{split}] base '{base}': present in {present}, missing in {missing}")

        # zbiorczo: różne rozmiary zbiorów baz
        sets = list(base_sets.values())
        if sets and not all(sets[0] == s for s in sets[1:]):
            sizes = ", ".join([f"{ch}={len(base_sets[ch])}" for ch in sorted(base_sets.keys())])
            errors.append(f"[{split}] base sets differ in size: {sizes}")

    return errors

# --- GUI pomocnicze ---
def show_scrollable(title, content):
    win = tk.Toplevel()
    win.title(title)
    win.geometry("1000x650")
    frm = ttk.Frame(win)
    frm.pack(fill="both", expand=True)
    yscroll = ttk.Scrollbar(frm, orient="vertical")
    xscroll = ttk.Scrollbar(frm, orient="horizontal")
    txt = tk.Text(frm, wrap="none", yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    yscroll.config(command=txt.yview)
    xscroll.config(command=txt.xview)
    yscroll.pack(side="right", fill="y")
    xscroll.pack(side="bottom", fill="x")
    txt.pack(side="left", fill="both", expand=True)
    txt.insert("1.0", content if content else "— brak treści —")
    txt.config(state="disabled")
    ttk.Button(win, text="OK", command=win.destroy).pack(pady=6)
    win.grab_set()
    win.wait_window()

# --- Aplikacja GUI ---
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("860x640")

        self.paths = {ch: tk.StringVar() for ch in CHANNELS}
        self.data_by_channel = None
        self._channel_errors = []
        self._cross_errors = []

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 5}

        frm_paths = ttk.LabelFrame(self, text="Ścieżki datasetów kanałowych (każdy ma images/ i labels/)")
        frm_paths.pack(fill="x", **pad)

        for ch in CHANNELS:
            row = ttk.Frame(frm_paths)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=ch, width=6).pack(side="left")
            ttk.Entry(row, textvariable=self.paths[ch]).pack(side="left", fill="x", expand=True)
            ttk.Button(row, text="Wybierz…", command=lambda c=ch: self.choose_dir(c)).pack(side="left", padx=6)

        ttk.Button(self, text="Waliduj", command=self.validate).pack(**pad)

        table_frame = ttk.LabelFrame(self, text="Podsumowanie (liczba poprawnych serii — baza nazwy)")
        table_frame.pack(fill="both", expand=False, **pad)
        cols = ("channel", "split", "series_count", "notes")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=10)
        headers = {"channel": "Channel", "split": "Split", "series_count": "Series (base)", "notes": "Uwagi"}
        widths = {"channel": 90, "split": 90, "series_count": 140, "notes": 460}
        anchors = {"channel": "center", "split": "center", "series_count": "e", "notes": "w"}
        for c in cols:
            self.tree.heading(c, text=headers[c])
            self.tree.column(c, width=widths[c], anchor=anchors[c])
        self.tree.pack(fill="both", expand=True, padx=6, pady=6)

        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(log_frame, height=10)
        self.log.pack(fill="both", expand=True, padx=6, pady=6)

        btns = ttk.Frame(self)
        btns.pack(fill="x", **pad)
        ttk.Button(btns, text="Błędy kanałów…", command=self.show_channel_errors).pack(side="left")
        ttk.Button(btns, text="Niespójności między kanałami…", command=self.show_cross_errors).pack(side="left", padx=8)
        ttk.Button(btns, text="Zamknij", command=self.destroy).pack(side="right")

    def choose_dir(self, ch):
        path = filedialog.askdirectory(title=f"Wybierz katalog kanału {ch}")
        if path:
            self.paths[ch].set(path)

    def log_write(self, s: str):
        self.log.insert("end", s + "\n")
        self.log.see("end")
        self.update_idletasks()

    def validate(self):
        self.log.delete("1.0", "end")
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._channel_errors = []
        self._cross_errors = []
        self.data_by_channel = {}

        # 1) walidacja wewnątrz kanałów
        for ch in CHANNELS:
            root = Path(self.paths[ch].get().strip())
            if not root.exists():
                messagebox.showerror(APP_TITLE, f"{ch}: katalog nie istnieje")
                return
            self.log_write(f"[INFO] Skanuję kanał {ch} -> {root}")
            data = scan_dataset(root)
            self.data_by_channel[ch] = data

            for split in SPLITS:
                n = len(data["series_base"][split])
                note = ""
                if not (root / "images" / split).exists() or not (root / "labels" / split).exists():
                    note = "brak images/labels"
                self.tree.insert("", "end", values=(ch, split, n, note))

            self._channel_errors.extend(data["errors"])

        # 2) porównanie między kanałami (po bazie nazwy)
        self.log_write("[INFO] Sprawdzam spójność między kanałami (po bazie nazwy)…")
        cross = compare_across_channels(self.data_by_channel)
        self._cross_errors = cross

        if self._channel_errors:
            self.log_write(f"[WARN] Błędy wewnątrz kanałów: {len(self._channel_errors)} (szczegóły w oknie)")

        if cross:
            self.log_write(f"[WARN] Niespójności między kanałami: {len(cross)} (szczegóły w oknie)")
            messagebox.showwarning(APP_TITLE, "Wykryto niespójności między kanałami.")
        else:
            self.log_write("[OK] Kanały spójne względem baz nazw.")
            if not self._channel_errors:
                messagebox.showinfo(APP_TITLE, "Dataset jest spójny i poprawny.")

    def show_channel_errors(self):
        show_scrollable("Błędy wewnątrz kanałów", "\n".join(self._channel_errors))

    def show_cross_errors(self):
        show_scrollable("Niespójności między kanałami", "\n".join(self._cross_errors))

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
