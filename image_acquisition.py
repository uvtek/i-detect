import pypylon.pylon as py
import numpy as np
import cv2
import threading
import time
import datetime
from model_config import MODEL_CONF

class ImageAcquisition:
    def __init__(self,mbClient,gui):
        self.SCANLINE_HEIGHT = 1
        self.VIRTUAL_FRAME_HEIGHT = 4800
        self.mbClient = mbClient
        self.gui = gui
        self.captured_image = None
        self.preprocessed_image = None
        self.camera_rate=0
        self.acqusition_speed_ratio=97.5
        self.direction= None

        self.model_name = None
        tlf = py.TlFactory.GetInstance()
        self.cam = py.InstantCamera(tlf.CreateFirstDevice())
        self.cam.Open()
        self.is_started = False
        self.is_in = False
        self.count = 0
        
    def clear_images(self):
        """Yakalanan ve işlenmiş görüntüleri temizle"""
        self.captured_image = None
        self.preprocessed_image = None

    def stop(self):
        self.mbClient.write_single_register(2, 0)
        self.mbClient.write_single_register(3, 0)
        self.direction=None
        self.gui.log("STOP")

    def update_speed_label(self):
        velocity=int(self.gui.speed_value.get())
        self.camera_rate=velocity*self.acqusition_speed_ratio
        self.gui.speed_value_label.config(text=f"{velocity}")
        self.mbClient.write_single_register(4, int(velocity*4.16))
    
    def set_parameter(self, model_name, camera_rate, gain=None, exposure=None, speed=None):
        self.model_name = model_name
        self.cam.Height.SetValue(self.SCANLINE_HEIGHT)
        self.cam.Width = self.cam.Width.Max
        self.cam.PixelFormat = "Mono8"
        self.speed = speed
        
        # Use provided gain and exposure if available, otherwise fall back to MODEL_CONF
        if gain is not None:
            self.cam.GainRaw = gain
        else:
            self.cam.GainRaw = MODEL_CONF[self.model_name]["gain"]
            
        if exposure is not None:
            self.cam.ExposureTimeAbs = exposure
        else:
            self.cam.ExposureTimeAbs = MODEL_CONF[self.model_name]["exposure"]
        
        self.cam.AcquisitionLineRateAbs.SetValue(camera_rate)
        
        print(f"CAMERA RATE: {camera_rate}")
        print(f"Resulting framerate: {self.cam.ResultingFrameRateAbs.Value}")

    def perform_capture_thread(self,selected_model,direction):
        self.update_speed_label()
        while True:
            if self.mbClient.read_holding_registers(7,1)[0]:
                self.set_parameter(selected_model,self.camera_rate)
                # self.capture_image_thread(direction)
                thread=threading.Thread(target=self.capture_image_thread(direction))
                thread.start()
                return True
            else:
                pass

    def capture_image_thread(self,direction):
        self.direction= direction
        start_time = time.time()
        self.captured_image = self.capture_image()
        end_time = time.time()
        self.gui.update_image(self.gui.original_label, self.captured_image)
        self.gui.log(f"Capture completed in {end_time - start_time:.2f} seconds.")
        self.stop()

    def capture_image(self):
        print(f"Using model: {self.model_name}, Gain: {self.cam.GainRaw.Value}, Exposure: {self.cam.ExposureTimeAbs.Value}")

        self.cam.StartGrabbing()
        img = np.ones((self.VIRTUAL_FRAME_HEIGHT, self.cam.Width.Value), dtype=np.uint8)
        missing_line = np.ones((self.SCANLINE_HEIGHT, self.cam.Width.Value), dtype=np.uint8) * 255

        for idx in range(self.VIRTUAL_FRAME_HEIGHT // self.SCANLINE_HEIGHT):
            with self.cam.RetrieveResult(1000, py.TimeoutHandling_ThrowException) as result:
                if result.GrabSucceeded():
                    out_array = result.GetArray().copy()
                    img[idx * self.SCANLINE_HEIGHT:idx * self.SCANLINE_HEIGHT + self.SCANLINE_HEIGHT] = out_array
                else:
                    print(f"index={idx}")
                    img[idx * self.SCANLINE_HEIGHT:idx * self.SCANLINE_HEIGHT + self.SCANLINE_HEIGHT] = missing_line

        self.cam.StopGrabbing()
        # self.cam.Close()

        #img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_normalized = img
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format as YYYY-MM-DD_HH-MM-SS
        # filename = f"images/image_{self.count}_exp_{self.cam.ExposureTimeAbs.Value}_gain{self.cam.GainRaw.Value}_{current_time}_speed_{self.speed}.jpg"
        filename = "images/raw_image.jpg"
        cv2.imwrite(filename, img_normalized)
        self.count += 1

        return img_normalized
    
    def test_set_parameters(self, model_config,speed_value):
        self.speed_value = speed_value
        self.update_speed_label()
        self.model_config = model_config

    def test_capture(self):
        """Test fonksiyonu"""
        test_config = {"gain": 256, "exposure": 256}
        self.test_set_parameters(test_config, 1387)
        img = self.test_capture_image()
        cv2.imwrite("images/test_capture.jpg", img)
        return "Test görüntüsü kaydedildi: test_capture.jpg"
    
    def test_capture_image(self,model_config,speed_value):
        self.test_set_parameters(model_config,speed_value)
        tlf = py.TlFactory.GetInstance()
        cam = py.InstantCamera(tlf.CreateFirstDevice())
        cam.Open()
        
        # Kamera ayarları
        cam.Height = self.SCANLINE_HEIGHT
        cam.Width = cam.Width.Max
        cam.PixelFormat = "Mono8"
        cam.GainRaw = self.model_config["gain"]
        cam.ExposureTimeAbs = self.model_config["exposure"]
        cam.AcquisitionLineRateAbs.SetValue(self.camera_rate)
        
        # Görüntü yakalama
        cam.StartGrabbing()
        img = np.ones((self.VIRTUAL_FRAME_HEIGHT, cam.Width.Value), dtype=np.uint8)
        missing_line = np.ones((self.SCANLINE_HEIGHT, cam.Width.Value), dtype=np.uint8) * 255
        
        for idx in range(self.VIRTUAL_FRAME_HEIGHT // self.SCANLINE_HEIGHT):
            with cam.RetrieveResult(1000, py.TimeoutHandling_ThrowException) as result:
                if result.GrabSucceeded():
                    img[idx * self.SCANLINE_HEIGHT:(idx + 1) * self.SCANLINE_HEIGHT] = result.GetArray()
                else:
                    img[idx * self.SCANLINE_HEIGHT:(idx + 1) * self.SCANLINE_HEIGHT] = missing_line
        
        cam.StopGrabbing()
        cam.Close()
        
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
if __name__ == "__main__":
    # Test kodu
    acquisition = ImageAcquisition()
    print(acquisition.test_capture()) 