import cv2
import numpy as np
from rembg import remove
import os
import matplotlib.pyplot as plt
import time

class ImagePreprocessor:
    def __init__(self,gui=None):
        self.gui = gui
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.SAVE_PATH = os.path.join(self.current_dir, "images", "input.png")
        
        # images klasörünü oluştur
        os.makedirs(os.path.join(self.current_dir, "images"), exist_ok=True)
        
    def process_image(self, image_array):
        """Görüntü ön işleme"""
        # Görüntüyü doğrudan işle, gereksiz dönüşümleri kaldır
        output = remove(cv2.imencode('.png', image_array)[1].tobytes())
        
        # NumPy işlemlerini optimize et
        output_image = cv2.imdecode(np.frombuffer(output, np.uint8), cv2.IMREAD_UNCHANGED)
        
        # Alpha kanalı işleme - daha verimli kontür bulma
        alpha = output_image[:,:,3]
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Tek seferde numpy işlemleri
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            cropped_output = output_image[y:y+h, x:x+w]
            
            # Vektörize edilmiş işlemler
            mask = cropped_output[:,:,3:] / 255.0
            result = np.where(mask == 1, 
                            cropped_output[:,:,:3], 
                            255 * (1 - mask) + cropped_output[:,:,:3] * mask
                            ).astype(np.uint8)
        else:
            result = image_array
            
        cv2.imwrite(self.SAVE_PATH, result)
        return result

    def perform_preprocess_thread(self,captured_image):
        self.gui.log("Starting preprocessing...")
        start_time = time.time()
        self.preprocessed_image = self.process_image(captured_image)
        end_time = time.time()
        # self.gui.after(0, self.gui.update_image, self.gui.cropped_label, self.preprocessed_image)
        self.gui.update_image(self.gui.cropped_label, self.preprocessed_image)
        self.gui.log(f"Preprocessing completed in {end_time - start_time:.2f} seconds.")
        return True

    def test_preprocess(self, test_image_path):
        """Test fonksiyonu"""
        print(f"Test görüntüsü yükleniyor: {test_image_path}")
        captured_image = cv2.imread(test_image_path)
        if captured_image is None:
            print("Hata: Görüntü yüklenemedi!")
            return "Görüntü yüklenemedi"
        
        print("Ön işleme başlıyor...")
        start_time = time.time()
        result = self.process_image(captured_image)
        end_time = time.time()
        print(f"Preprocessing completed in {end_time - start_time:.2f} seconds.")
        
        # İşlenmiş görüntüyü kaydet
        output_path = os.path.join(self.current_dir, "images", "input_test.png")
        cv2.imwrite(output_path, result)
        print(f"İşlenmiş görüntü kaydedildi: {output_path}")

        # Matplotlib ile görüntüleme
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB))
        plt.title('Orijinal Görüntü')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('İşlenmiş Görüntü')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return "Test tamamlandı"

if __name__ == "__main__":
    # Test kodu
    print("Test başlıyor...")
    preprocessor = ImagePreprocessor()
    test_image = "images/raw_image.jpg"  # Test için görüntü yolunu belirtin
    print(preprocessor.test_preprocess(test_image)) 