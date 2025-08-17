import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import customtkinter as ctk
import tifffile  # pip install tifffile

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


class YOLOViewer:
    def __init__(self, master):
        self.master = master
        self.master.title("FrogPix")
        self.master.geometry("800x600")

        self.bg_image_path = r"C:\Users\topgu\Desktop\Splotowe Sieci Neuronowe\frogpix.png"
        self.icon_path     = r"C:\Users\topgu\Desktop\Splotowe Sieci Neuronowe\frogpix.ico"

        try:
            self.master.iconbitmap(self.icon_path)
        except Exception as e:
            print("Błąd podczas ładowania ikony:", e)

        try:
            bg = Image.open(self.bg_image_path).convert("RGB")
            self.bg_photo = ImageTk.PhotoImage(bg)
            lbl = tk.Label(master, image=self.bg_photo, bg='black')
            lbl.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception as e:
            print("Błąd przy ładowaniu tła:", e)

        self.image_paths = []
        self.current_index = 0
        font = ("Orbitron", 14)

        # główny widget obrazu
        self.image_label = tk.Label(master)
        self.image_label.pack(pady=10)

        # przyciski nawigacji
        self.nav_frame = tk.Frame(master, bg="#242424")
        self.btn_prev = ctk.CTkButton(self.nav_frame, text="<< Previous",
                                      command=self.show_prev_image,
                                      fg_color="#8A2BE2", text_color="white",
                                      hover_color="#7A1BBE", corner_radius=0,
                                      border_width=2, border_color="#7A1BBE",
                                      font=font, state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, padx=10)

        self.btn_next = ctk.CTkButton(self.nav_frame, text="Next >>",
                                      command=self.show_next_image,
                                      fg_color="#8A2BE2", text_color="white",
                                      hover_color="#7A1BBE", corner_radius=0,
                                      border_width=2, border_color="#7A1BBE",
                                      font=font, state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=10)
        self.nav_frame.pack_forget()

        # etykieta z nazwą obrazu i licznikiem
        self.info_label = ctk.CTkLabel(master, text="", font=font, text_color="white")

        # wybór folderu
        self.path_frame = tk.Frame(master, bg="#242424")
        self.input_entry = ctk.CTkEntry(self.path_frame, width=300,
                                        placeholder_text="Input folder",
                                        fg_color="black", text_color="white",
                                        font=font)
        self.input_entry.pack(side=tk.LEFT, padx=10)
        self.btn_load = ctk.CTkButton(self.path_frame, text="Select",
                                      command=self.select_folder,
                                      fg_color="#8A2BE2", text_color="white",
                                      hover_color="#7A1BBE", corner_radius=0,
                                      border_width=2, border_color="#7A1BBE",
                                      font=font)
        self.btn_load.pack(side=tk.LEFT)
        self.path_frame.pack(pady=10)

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select folder with images")
        if not folder:
            return

        self.image_paths = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]
        print("Znalezione pliki:", self.image_paths)
        if not self.image_paths:
            messagebox.showerror("Error", "No images found in the folder!")
            return

        self.current_index = 0
        state = tk.NORMAL if len(self.image_paths) > 1 else tk.DISABLED
        self.btn_next.configure(state=state)
        self.btn_prev.configure(state=state)

        self.master.configure(bg="#242424")
        # ukryj tło
        for w in self.master.place_slaves():
            if isinstance(w, tk.Label) and getattr(w, "image", None)==self.bg_photo:
                w.place_forget()

        self.nav_frame.pack(pady=10)
        self.info_label.pack(pady=10)
        self.display_current_image()

    def display_current_image(self):
        path = self.image_paths[self.current_index]
        ext = os.path.splitext(path)[1].lower()

        # 1) Wczytaj dane
        try:
            if ext in ('.tif', '.tiff'):
                arr = tifffile.imread(path)           # numpy array
                # wybierz RGB, lub zbuduj z jednokanałowego
                if arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)
                elif arr.ndim == 3 and arr.shape[2] > 3:
                    arr = arr[..., :3]
                # skalowanie do uint8, jeśli potrzeba
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.float32)
                    arr -= arr.min()
                    arr /= arr.max() if arr.max()>0 else 1
                    arr *= 255
                    arr = arr.astype(np.uint8)
                image = Image.fromarray(arr)
            else:
                image = Image.open(path).convert("RGB")
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Error", f"Cannot load image:\n{path}\n{e}")
            return

        # 2) Wczytaj YOLO-labelki
        lbl_file = os.path.splitext(path)[0] + ".txt"
        labels = []
        if os.path.exists(lbl_file):
            try:
                with open(lbl_file) as f:
                    labels = [L.strip() for L in f if L.strip()]
            except Exception as e:
                messagebox.showerror("Error", f"Cannot load labels:\n{lbl_file}\n{e}")

        # 3) Narysuj bounding boxy
        img_w, img_h = image.size
        draw = ImageDraw.Draw(image)
        for line in labels:
            parts = line.split()
            if len(parts) < 5: continue
            cid, xc, yc, wn, hn = parts[0], *map(float, parts[1:5])
            cx, cy = xc*img_w, yc*img_h
            bw, bh = wn*img_w, hn*img_h
            tl = (cx-bw/2, cy-bh/2)
            br = (cx+bw/2, cy+bh/2)
            draw.rectangle([tl, br], outline="lime", width=2)
            draw.text((tl[0], tl[1]-12), cid, fill="lime")

        # 4) Thumbnail + wyświetlenie
        image.thumbnail((800,600))
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.photo)

        # 5) Aktualizacja etykiety
        self.info_label.configure(
            text=f"{os.path.basename(path)} ({self.current_index+1}/{len(self.image_paths)})"
        )

    def show_next_image(self):
        self.current_index = (self.current_index+1) % len(self.image_paths)
        self.display_current_image()

    def show_prev_image(self):
        self.current_index = (self.current_index-1) % len(self.image_paths)
        self.display_current_image()


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    app = YOLOViewer(root)
    root.mainloop()
