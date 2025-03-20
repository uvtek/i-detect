import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk, ImageDraw
import docker
import os
import numpy as np

class GUI(tk.Tk):
    def __init__(self, container):
        super().__init__()
        
        self.container = container
        self.DISPLAY_WIDTH = 300
        self.DISPLAY_HEIGHT = 300
        self.RESULT_DISPLAY_WIDTH = self.DISPLAY_WIDTH * 1
        self.is_on = True

        self.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.title('Image Viewer')
        self.geometry('1500x500')

        # Load button images
        self.on_image = ImageTk.PhotoImage(Image.open("assets/off.jpg").resize((50, 50), Image.LANCZOS))
        self.off_image = ImageTk.PhotoImage(Image.open("assets/on.jpg").resize((50, 50), Image.LANCZOS))
        self.sag_image = ImageTk.PhotoImage(Image.open("assets/sag.jpg").rotate(-90).resize((50, 50), Image.LANCZOS))
        self.sol_image = ImageTk.PhotoImage(Image.open("assets/sol.jpg").rotate(-90).resize((50, 50), Image.LANCZOS))
        self.reset_image = ImageTk.PhotoImage(Image.open("assets/reset.jpg").resize((50, 50), Image.LANCZOS))
        self.stop_image = ImageTk.PhotoImage(Image.open("assets/stop.png").resize((50, 50), Image.LANCZOS))

        self.create_widgets()
        
    def create_widgets(self):
        self.menu_frame = ttk.Frame(self)
        self.menu_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        self.reset_button = ttk.Button(self.menu_frame, image=self.reset_image)
        self.reset_button.grid(row=0, column=2, pady=5)

        self.on_off_button = ttk.Button(self.menu_frame, image=self.on_image)
        self.on_off_button.grid(row=0, column=0, pady=5)

        self.stop_button = ttk.Button(self.menu_frame, image=self.stop_image)
        self.stop_button.grid(row=0, column=1, pady=5)

        self.sol_button = ttk.Button(self.menu_frame, image=self.sol_image)
        self.sol_button.grid(row=1, column=0)

        self.sag_button = ttk.Button(self.menu_frame, image=self.sag_image)
        self.sag_button.grid(row=1, column=2)

        # Speed control
        self.speed_label = ttk.Label(self.menu_frame, text="Hız:", font=("Arial", 12))
        self.speed_label.grid(row=2, column=1, pady=5)

        self.speed_value = tk.IntVar(self, 30)  # Default speed value
        self.speed_slider = ttk.Scale(self.menu_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                    variable=self.speed_value)
        self.speed_slider.grid(row=3, column=0, columnspan=3, pady=5)

        self.speed_value_label = ttk.Label(self.menu_frame, text=f"{self.speed_value.get()}", 
                                         font=("Arial", 12))
        self.speed_value_label.grid(row=4, column=1, pady=5)

        # Image display areas
        self.empty_image = self.create_empty_image()
        self.empty_result_image = self.create_empty_image(width=self.RESULT_DISPLAY_WIDTH)
        
        self.original_label = ttk.Label(self, image=self.empty_image)
        self.cropped_label = ttk.Label(self, image=self.empty_image)
        self.result_label = ttk.Label(self, image=self.empty_result_image)
        
        self.original_label.grid(row=0, column=1, padx=10, pady=10)
        self.cropped_label.grid(row=0, column=2, padx=10, pady=10)
        self.result_label.grid(row=0, column=3, padx=10, pady=10)

        # Configure grid
        for i in range(6):
            self.grid_columnconfigure(i, weight=1)

        # Model selection
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(self, textvariable=self.model_var, 
                                         state="readonly", width=50)
        self.model_dropdown.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="w")

        # Gain and Exposure display
        self.gain_var = tk.StringVar()
        self.exposure_var = tk.StringVar()

        gain_container = ttk.Frame(self)
        gain_container.grid(row=1, column=1)
        ttk.Label(gain_container, text="Gain:").pack(side="left", padx=(0, 5))
        ttk.Label(gain_container, textvariable=self.gain_var).pack(side="left")

        exposure_container = ttk.Frame(self)
        exposure_container.grid(row=1, column=1, sticky="e")
        ttk.Label(exposure_container, text="Exposure:").pack(side="left", padx=(0, 5))
        ttk.Label(exposure_container, textvariable=self.exposure_var).pack(side="left")

        # Log window
        self.log_window = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=60, height=10)
        self.log_window.grid(row=3, column=0, columnspan=6, padx=10, pady=10, sticky="nsew")

    def create_empty_image(self, width=None, height=None):
        if width is None:
            width = self.DISPLAY_WIDTH
        if height is None:
            height = self.DISPLAY_HEIGHT
            
        img = Image.new('RGB', (width, height), color='lightgray')
        d = ImageDraw.Draw(img)
        d.text((width//2-30, height//2), "No Image", fill=(0, 0, 0))
        return ImageTk.PhotoImage(img)

    def update_image(self, label, image, width=None, height=None):
        if width is None:
            width = self.DISPLAY_WIDTH
        if height is None:
            height = self.DISPLAY_HEIGHT
            
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = image.resize((width, height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        
        label.config(image=photo)
        label.image = photo

    def log(self, message):
        self.log_window.insert(tk.END, message + "\n")
        self.log_window.see(tk.END)

    def clear_images(self):
        """Görüntü alanlarını temizle"""
        self.original_label.config(image=self.empty_image)
        self.original_label.image = self.empty_image
        self.cropped_label.config(image=self.empty_image)
        self.cropped_label.image = self.empty_image
        self.result_label.config(image=self.empty_result_image)
        self.result_label.image = self.empty_result_image

    def update_displays(self, original, preprocessed, result):
        """Tüm görüntü alanlarını güncelle"""
        if original is not None:
            self.update_image(self.original_label, original)
        if preprocessed is not None:
            self.update_image(self.cropped_label, preprocessed)
        if result is not None:
            self.update_image(self.result_label, result, width=self.RESULT_DISPLAY_WIDTH)

    def on_exit(self):
        self.destroy()

if __name__ == "__main__":
    # Docker client oluşturuluyor
    client = docker.from_env()
    # Triton Inference Server konteyneri çalıştırılıyor
    container = client.containers.run(
        "my_tritonserver",  # Yeni imaj adı
        tty=True,
        stdin_open=True,
        network_mode="host",
        name="Triton_Inference_Server",
        volumes={
            "/home/nvidia/Documents/Argemtek_AGX/triton_inference_server/i-detect/clients": {
                "bind": "/workspace/clients", "mode": "rw"
            },
            "/home/nvidia/Documents/Argemtek_AGX/triton_inference_server/i-detect/images": {
                "bind": "/workspace/images", "mode": "rw"
            }
        },
        detach=True
    )
    
    try:
        app = GUI(container)
        app.mainloop()
    finally:
        # GUI kapandıktan sonra container'ı temizle
        try:
            container.stop()
            container.remove()
            print("Container başarıyla kapatıldı ve kaldırıldı.")
        except Exception as e:
            print(f"Container kapatılırken hata oluştu: {str(e)}") 