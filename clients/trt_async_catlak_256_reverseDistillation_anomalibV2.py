import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tritonclient.grpc.aio as grpcclient
import asyncio
import time
import json


def preprocess(img_array):
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    return img_array

def postprocess(defect_image):
    return defect_image.squeeze()

def split_image(image, tile_size=(512, 512)):
    width, height = image.size
    tiles = []
    for i in range(0, height, tile_size[1]):
        for j in range(0, width, tile_size[0]):
            box = (j, i, min(j+tile_size[0], width), min(i+tile_size[1], height))
            tile = image.crop(box)
            tile_array = np.array(tile)
            if tile_array.shape[2] == 4:
                tile_array = tile_array[:, :, :3]
            padded_tile = np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
            padded_tile[:tile_array.shape[0], :tile_array.shape[1], :] = tile_array
            tiles.append((Image.fromarray(padded_tile), box))
    return tiles

async def infer_tile(triton_client, model_name, tile):
    tile_array = np.array(tile)
    input_data = preprocess(tile_array)
    input_data = np.expand_dims(input_data, axis=0)
    
    inputs = [grpcclient.InferInput('input', input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)
    
    outputs = [grpcclient.InferRequestedOutput('output')]
    
    result = await triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    
    return postprocess(result.as_numpy('output'))

def optimize_threshold(params):
    # Piksel seviyesi threshold'u kullanıyoruz, çünkü daha hassas
    base_threshold = params['pixel_threshold']
    
    # Anomali haritasının min ve max değerlerini kullanarak normalize ediyoruz
    anomaly_min = params['anomaly_maps.min']
    anomaly_max = params['anomaly_maps.max']
    
    # Normalize edilmiş threshold hesaplama
    normalized_threshold = (base_threshold - anomaly_min) / (anomaly_max - anomaly_min)
    
    # Güvenlik için sınırlama
    return max(0.05, min(normalized_threshold, 0.95))

async def infer(model_name, image_path, url, params):
    threshold = optimize_threshold(params)
    
    try:
        async with grpcclient.InferenceServerClient(url=url) as triton_client:
            original_image = Image.open(image_path).convert('RGB')
            width, height = original_image.size
            
            tiles = split_image(original_image)
            
            result = np.zeros((height, width), dtype=np.float32)
            
            tasks = [infer_tile(triton_client, model_name, tile[0]) for tile in tiles]
            tile_results = await asyncio.gather(*tasks)
            
            for (tile, box), tile_result in zip(tiles, tile_results):
                x1, y1, x2, y2 = box
                result[y1:y2, x1:x2] = tile_result[:y2-y1, :x2-x1]
            
            # Normalize result using the anomaly map min and max from params
            result_normalized = (result - params['anomaly_maps.min']) / (params['anomaly_maps.max'] - params['anomaly_maps.min'])
            #result_normalized = np.clip(result_normalized, 0, 1)
            
            max_anomaly_score = np.max(result_normalized)
            has_defect = max_anomaly_score > threshold
            
            # Threshold the result
            result_thresholded = (result_normalized > threshold).astype(np.uint8) * 255
            
            # Create output image
            output_image = Image.fromarray(result_thresholded, mode='L')
            draw = ImageDraw.Draw(output_image)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            except IOError:
                font = ImageFont.load_default()
            
            text_color = 255  # White color for text
            
            draw.text((10, 10), f"Defect: {'Yes' if has_defect else 'No'}", font=font, fill=text_color)
            draw.text((10, 50), f"Max anomaly score: {max_anomaly_score:.4f}", font=font, fill=text_color)
            draw.text((10, 90), f"Optimized threshold: {threshold:.4f}", font=font, fill=text_color)
            
            output_image.save('/workspace/images/result_annotated.png')
            result_thresholded.save('/workspace/images/result_thresholded.png')

            return result, has_defect, max_anomaly_score, threshold
            
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--url', default='localhost:8001', help='Inference server URL')
    parser.add_argument('--params', required=True, help='JSON file path for model parameters')
    args = parser.parse_args()

    with open(args.params, 'r') as f:
        params = json.load(f)

    start_time = time.time()
    result, has_defect, max_anomaly_score, optimized_threshold = await infer(args.model, args.image, args.url, params)
    total_time = time.time() - start_time

    print(f"Processing completed. Check '/workspace/images/result_thresholded.png' for the result.")
    print(f"Defect: {'Yes' if has_defect else 'No'}")
    print(f"Max anomaly score: {max_anomaly_score:.4f}")
    print(f"Optimized threshold: {optimized_threshold:.4f}")
    print(f"Total processing time: {total_time:.2f} seconds")

if __name__ == '__main__':
    asyncio.run(main())