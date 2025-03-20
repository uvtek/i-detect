INFERENCE SERVER CONTAINER

docker run --gpus=1 --rm --net=host -v /home/nvidia/Documents/Argemtek_AGX/triton_inference_server/model_repository:/models nvcr.io/nvidia/tritonserver:24.06-py3-igpu tritonserver --model-repository=/models


SDK CONTAINER

docker run -it --rm --net=host -v /home/nvidia/Documents/Argemtek_AGX/triton_inference_server/i-detect/clients:/workspace/clients  -v /home/nvidia/Documents/Argemtek_AGX/triton_inference_server/i-detect/images:/workspace/images nvcr.io/nvidia/tritonserver:24.06-py3-sdk
pip install opencv-python-headless

docker run -it --rm --net=host -v /home/nvidia/Documents/Argemtek_AGX/triton_inference_server/i-detect/clients:/workspace/clients -v /home/nvidia/Documents/Argemtek_AGX/triton_inference_server/i-detect/images:/workspace/images my_tritonserver

TRT CONVERT

docker run -it --rm --gpus all -v /home/nvidia/Documents/Argemtek_AGX/triton_inference_server:/workspace nvcr.io/nvidia/tritonserver:24.06-py3-igpu bash
docker exec -it 659 bash


cd /workspace/
/usr/src/tensorrt/bin/trtexec --onnx=/workspace/model.onnx --saveEngine=/workspace/model.engine --fp16

find / -name trtexec 2>/dev/null

/usr/src/tensorrt/bin/trtexec --onnx=/workspace/model.onnx --saveEngine=/workspace/model.engine --int8 --minShapes=input:1x3x512x512 --optShapes=input:16x3x512x512 --maxShapes=input:16x3x512x512 --threads --verbose
