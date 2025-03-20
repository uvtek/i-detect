import docker
import time
from model_config import MODEL_CONF
import subprocess
from PIL import Image

class TritonInference:
    def __init__(self, gui=None,container=None):
        self.gui = gui
        self.client = docker.from_env()
        self.container = container
        self.container_name = "Triton_Inference_Server"
        self.RESULT_PATH = "/home/nvidia/Documents/Argemtek_AGX/triton_inference_server/i-detect/images/result_thresholded.png"

    def perform_inference_thread(self,selected_model):
        self.gui.log("Starting inference...")
        start_time = time.time()
        self.run_inference(selected_model)
        end_time = time.time()
        result_img = Image.open(self.RESULT_PATH)
        self.gui.update_image(self.gui.result_label, result_img)
        self.gui.log(f"Preprocessing completed in {end_time - start_time:.2f} seconds.")
        #self.start_button.config(state='normal')
        return True

    def run_inference(self,model_name):
        print(f"Running inference with model: {model_name}")
        inference_command = [
            "docker", "run", "--rm", "--net=host",
            "-v", "/home/nvidia/Documents/Argemtek_AGX/triton_inference_server/i-detect/clients:/workspace/clients",
            "-v", "/home/nvidia/Documents/Argemtek_AGX/triton_inference_server/i-detect/images:/workspace/images",
            "-v", "/home/nvidia/Documents/Argemtek_AGX/triton_inference_server/model_repository:/workspace/model_repository",
            "nvcr.io/nvidia/tritonserver:24.06-py3-sdk",
            "python", "/workspace/clients/trt_async_catlak_256_reverseDistillation_anomalibV3.py",
            "--model", model_name,
            "--threshold", str(MODEL_CONF[model_name]["threshold"]),
            # "--params", f"/workspace/model_repository/{model_name}/metadata.json",
            "--image", "/workspace/images/input.png",
            "--url", "localhost:8001"
        ]

        result = subprocess.run(inference_command, capture_output=True, text=True)
        # print("Inference result:", result.stderr)

    def test_prepare_container(self):
        self.client = docker.from_env()
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

    def test_perform_inference_thread(self):
        print("Starting inference...")
        start_time = time.time()
        self.test_inference()
        end_time = time.time()
        result_img = Image.open(self.RESULT_PATH)
        print(f"Preprocessing completed in {end_time - start_time:.2f} seconds.")
        return True

    def test_inference(self):
        model_name="trt_white_512_dynamic_reverseDistillation_anomalib"
        print(f"Running inference with model: {model_name}")
        inference_command = [
            "docker", "run", "--rm", "--net=host",
            "-v", "/home/nvidia/Documents/Argemtek_AGX/triton_inference_server/i-detect/clients:/workspace/clients",
            "-v", "/home/nvidia/Documents/Argemtek_AGX/triton_inference_server/i-detect/images:/workspace/images",
            "-v", "/home/nvidia/Documents/Argemtek_AGX/triton_inference_server/model_repository:/workspace/model_repository",
            "nvcr.io/nvidia/tritonserver:24.06-py3-sdk",
            "python", "/workspace/clients/trt_async_catlak_256_reverseDistillation_anomalibV3.py",
            "--model", model_name,
            "--threshold", str(MODEL_CONF[model_name]["threshold"]),
            # "--params", f"/workspace/model_repository/{model_name}/metadata.json",
            "--image", "/workspace/images/input_test.png",
            "--url", "localhost:8001"
        ]

        result = subprocess.run(inference_command, capture_output=True, text=True)

if __name__ == "__main__":

    inference = TritonInference()
    inference.test_prepare_container()
    print(inference.test_perform_inference_thread()) 