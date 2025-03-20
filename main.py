from gui_interface import GUI
from image_acquisition import ImageAcquisition
from image_preprocessing import ImagePreprocessor
from triton_inference import TritonInference
from pyModbusTCP.client import ModbusClient
from model_config import MODEL_CONF
import docker
import threading

class MainApplication:
    def __init__(self):
        self.direction = None
        self.acqusition_speed_ratio = 97.5
        self.default_velocity = 30  # Varsayılan hız değeri
        self.camera_rate = self.default_velocity * self.acqusition_speed_ratio  # Başlangıç kamera hızı
        self.vel = self.default_velocity  # Varsayılan hız değeri
        self.mbClient = ModbusClient(host='192.168.68.10', port=502, auto_open=True)
        
        # Docker client'ı başlat ve container'ı hazırla
        self.client = docker.from_env()
        self.prepare_container()
        
        # Container'ı GUI'ye ve TritonInference'a geç
        self.gui = GUI(self.container)
        self.acquisition = ImageAcquisition(self.mbClient,self.gui)
        self.preprocessor = ImagePreprocessor(self.gui)
        self.inference = TritonInference(self.gui,self.container)
        
        self.setup_callbacks()
                    
    def run(self):
        # Sonra GUI'yi başlat
        self.gui.mainloop()
        
    def prepare_container(self):
        """Container'ı hazırla ve başlat"""
        try:
            # Eski container'ı temizle
            containers = self.client.containers.list(all=True)
            for container in containers:
                if container.name == "Triton_Inference_Server":
                    container.remove(force=True)
                    print("Eski container temizlendi")
                    break
            
            # Yeni container'ı oluştur
            self.container = self.client.containers.run(
                "my_tritonserver",
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
            print("Yeni container oluşturuldu")
            
        except Exception as e:
            print(f"Container hazırlama hatası: {str(e)}")
            raise
    
    def cleanup(self):
        """Uygulama kapanırken kaynakları temizle"""
        try:
            # Inference cleanup
            if hasattr(self, 'inference'):
                self.inference.cleanup()
            
            # Container cleanup
            if hasattr(self, 'container'):
                try:
                    self.container.stop()
                    self.container.remove()
                    print("Container başarıyla kapatıldı ve kaldırıldı.")
                except Exception as e:
                    print(f"Container kapatılırken hata oluştu: {str(e)}")
            
            # ModbusTCP bağlantısını kapat
            if hasattr(self, 'mbClient'):
                self.mbClient.close()
                print("ModbusTCP bağlantısı kapatıldı.")
                
        except Exception as e:
            print(f"Cleanup sırasında hata oluştu: {str(e)}")
    
    def setup_callbacks(self):
        """GUI olaylarını işleyicilere bağla"""
        # Buton callback'leri
        self.gui.on_off_button.config(command=self.toggle_power)
        self.gui.sag_button.config(command=self.forward)
        self.gui.sol_button.config(command=self.backward)
        self.gui.stop_button.config(command=self.stop)
        self.gui.reset_button.config(command=self.reset)
        
        # Model seçimi ve hız kontrolü
        self.gui.model_dropdown.bind("<<ComboboxSelected>>", self.on_model_selected)
        self.gui.speed_slider.config(command=self.update_speed_label)
        
        # Model listesini doldur
        self.gui.model_dropdown['values'] = list(MODEL_CONF.keys())
        if self.gui.model_dropdown['values']:
            self.gui.model_dropdown.set(self.gui.model_dropdown['values'][0])
            self.on_model_selected(None)  # İlk model seçimini yap
            
    def toggle_power(self):
        if self.gui.is_on:
            self.mbClient.write_single_register(0, 1)
            self.gui.log("POWER ON")
            self.gui.on_off_button.config(image=self.gui.off_image)
        else:
            self.mbClient.write_single_register(0, 0)
            self.gui.log("POWER OFF")
            self.gui.on_off_button.config(image=self.gui.on_image)
        self.gui.is_on = not self.gui.is_on

    def stop(self):
        self.mbClient.write_single_register(2, 0)
        self.mbClient.write_single_register(3, 0)
        self.direction = None
        self.gui.log("STOP")

    def reset(self):
        self.mbClient.write_single_register(6, 1)
        self.gui.log("RESET")

    def update_speed_label(self, event):
        velocity = int(self.gui.speed_value.get())
        self.vel = velocity
        self.camera_rate = velocity * self.acqusition_speed_ratio
        self.gui.speed_value_label.config(text=f"{velocity}")
        self.mbClient.write_single_register(4, int(velocity * 4.16))
        
    def on_model_selected(self, event):
        self.selected_model = self.gui.model_var.get()
        if self.selected_model in MODEL_CONF:
            self.gui.gain_var.set(str(MODEL_CONF[self.selected_model]["gain"]))
            self.gui.exposure_var.set(str(MODEL_CONF[self.selected_model]["exposure"]))
            self.gui.log(f"Model seçildi: {self.selected_model}")
    
    def backward(self):
        if not self.gui.is_on:
            if not self.direction or self.direction is None:
                self.mbClient.write_single_register(3, 1)
                self.direction = False
                self.gui.log("BACKWARD")
            else:
                self.gui.log("PLEASE FIRSTLY STOP")
        else:
            self.gui.log("PLEASE ENABLE POWER")
    
    def forward(self):
        if not self.gui.is_on:
            if self.direction or self.direction is None:
                self.mbClient.write_single_register(2, 1)
                self.direction = True
                self.gui.log("FORWARD")
                thread=threading.Thread(target=self.start_process)
                thread.start()
            else:
                self.gui.log("PLEASE FIRSTLY STOP")
        else:
            self.gui.log("PLEASE ENABLE POWER")

    def start_process(self):
        """Görüntü işleme sürecini başlat"""
        self.gui.clear_images()
        self.gui.update_idletasks()
        
        # Görüntü yakalama
        self.gui.log("Görüntü yakalama başladı...")
        captured_image = self.acquisition.perform_capture_thread(self.selected_model,self.direction)
        if captured_image is None:
            self.gui.log("Görüntü yakalama başarısız!")
            return
            
        # Görüntü ön işleme
        self.gui.log("Görüntü ön işleme başladı...")
        preprocessed_image = self.preprocessor.perform_preprocess_thread(self.acquisition.captured_image)
        if preprocessed_image is None:
            self.gui.log("Görüntü ön işleme başarısız!")
            return
            
        # Triton Inference
        self.gui.log("Triton Inference başladı...")
        result_image = self.inference.perform_inference_thread(self.selected_model)
        if result_image is None:
            self.gui.log("Triton Inference başarısız!")
            return
            
        self.gui.log("İşlem tamamlandı!")

if __name__ == "__main__":
    app = MainApplication()
    try:
        app.run()
    finally:
        app.cleanup()
