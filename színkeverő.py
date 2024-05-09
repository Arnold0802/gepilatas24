import tkinter as tk
from tkinter import Scale, Label
from PIL import Image, ImageTk

class ColorMixer:
    def __init__(self, master):
        self.master = master
        master.title("Színkeverő alkalmazás")

        # Címkék és csúszkák inicializálása
        self.b = Scale(master, from_=0, to=255, orient="horizontal", label="Kék")
        self.b.pack()
        self.g = Scale(master, from_=0, to=255, orient="horizontal", label="Zöld")
        self.g.pack()
        self.r = Scale(master, from_=0, to=255, orient="horizontal", label="Vörös")
        self.r.pack()

        # Szín megjelenítő inicializálása
        self.color_display = Label(master, text="         ", bg="#000000", height=9, width=40)
        self.color_display.pack(pady=20)

        # Frissítés a csúszkák mozgatásakor
        self.b.bind("<B1-Motion>", self.update_color)
        self.g.bind("<B1-Motion>", self.update_color)
        self.r.bind("<B1-Motion>", self.update_color)

    def update_color(self, event=None):
        """Frissíti a címke háttérszínét az RGB csúszkák állása alapján."""
        b, g, r = self.r.get(), self.g.get(), self.b.get()
        color = f"#{b:02x}{g:02x}{r:02x}"
        self.color_display.config(bg=color)
        #self.color_display.config(text=color)  # Opcionális: szöveg megjelenítése a színhez

# GUI indítása
root = tk.Tk()
app = ColorMixer(root)
root.mainloop()
