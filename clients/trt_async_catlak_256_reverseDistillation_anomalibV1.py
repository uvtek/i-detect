import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tritonclient.grpc.aio as grpcclient
import asyncio
import time
import matplotlib.pyplot as plt

def preprocess(img_array):
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    return img_array

def postprocess(defect_image):
    defect_image = defect_image.squeeze()
    return defect_image

def split_image(image, tile_size=(512, 512)):
    width, height = image.size
    tiles = []
    for i in range(0, height, tile_size[1]):
        for j in range(0, width, tile_size[0]):
            box = (j, i, min(j+tile_size[0], width), min(i+tile_size[1], height))
            tile = image.crop(box)
            tile_array = np.array(tile)
            
            # Convert RGBA to RGB if necessary
            if tile_array.shape[2] == 4:
                tile_array = tile_array[:, :, :3]
            
            padded_tile = np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
            padded_tile[:tile_array.shape[0], :tile_array.shape[1], :] = tile_array
            tiles.append((Image.fromarray(padded_tile), box))
    return tiles

async def infer_tile(triton_client, model_name, tile):
    tile_array = np.array(tile)
    input_data = preprocess(tile_array)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    
    inputs = []
    inputs.append(grpcclient.InferInput('input', input_data.shape, "FP32"))
    inputs[0].set_data_from_numpy(input_data)
    
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('output'))
    
    result = await triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    
    output = result.as_numpy('output')
    return postprocess(output)

def apply_colormap(image, cmap_name='jet'):
    cmap = plt.get_cmap(cmap_name)
    colored_image = cmap(image)
    return (colored_image[:, :, :3] * 255).astype(np.uint8)

def calculate_anomaly_score(result_normalized, edge_percent=0.1):
    height, width = result_normalized.shape
    edge_h = int(height * edge_percent)
    edge_w = int(width * edge_percent)
    
    # Create a mask to ignore edge regions
    mask = np.ones_like(result_normalized, dtype=bool)
    mask[:edge_h, :] = False
    mask[-edge_h:, :] = False
    mask[:, :edge_w] = False
    mask[:, -edge_w:] = False
    
    # Calculate max anomaly score only for non-edge regions
    max_anomaly_score = np.max(result_normalized[mask])
    
    return max_anomaly_score

def draw_text_with_background(draw, text, position, font, text_color, bg_color):
    # Get the bounding box of the text
    bbox = draw.textbbox(position, text, font=font)
    
    # Draw the background rectangle
    draw.rectangle(bbox, fill=bg_color)
    
    # Draw the text
    draw.text(position, text, font=font, fill=text_color)

async def infer(model_name, image_path, url, cmap_name='jet', threshold=0.5):
    try:
        async with grpcclient.InferenceServerClient(url=url) as triton_client:
            # Load the image and convert to RGB
            original_image = Image.open(image_path).convert('RGB')
            width, height = original_image.size
            
            # Split the image into tiles
            tiles = split_image(original_image)
            
            # Create an empty array to store the result
            result = np.zeros((height, width), dtype=np.float32)
            
            # Process each tile asynchronously
            tasks = [infer_tile(triton_client, model_name, tile[0]) for tile in tiles]
            tile_results = await asyncio.gather(*tasks)
            
            # Place the results in the correct positions
            for (tile, box), tile_result in zip(tiles, tile_results):
                x1, y1, x2, y2 = box
                result[y1:y2, x1:x2] = tile_result[:y2-y1, :x2-x1]
            
            # Normalize the result to 0-1 range
            result_normalized = (result - result.min()) / (result.max() - result.min())
            
            # Calculate max anomaly score ignoring edge regions
            max_anomaly_score = calculate_anomaly_score(result_normalized)
            has_defect = max_anomaly_score > threshold
            
            # Apply colormap
            result_colored = apply_colormap(result_normalized, cmap_name)
            
            # Create a new image combining original and result
            combined_image = Image.new('RGB', (width * 2, height))
            combined_image.paste(original_image, (0, 0))
            combined_image.paste(Image.fromarray(result_colored), (width, 0))
            
            # Add text to the image with larger font and background
            draw = ImageDraw.Draw(combined_image)
            
            # Load a larger font (increase size by 600% from the original)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)  # Increased from 36 to 72
            except IOError:
                font = ImageFont.load_default().font.copy()
                font.set_size(72)
            
            text_color = (255, 255, 255)  # White color for text
            bg_color = (0, 0, 0)  # Black color for background
            
            draw_text_with_background(draw, f"Defect: {'Yes' if has_defect else 'No'}", (60, 60), font, text_color, bg_color)
            draw_text_with_background(draw, f"Max anomaly score: {max_anomaly_score:.4f}", (60, 180), font, text_color, bg_color)
            draw_text_with_background(draw, f"Threshold: {threshold}", (60, 300), font, text_color, bg_color)
            
            # # Save the combined image
            combined_image.save('/workspace/images/result_annotated.png')
            # Image.fromarray(result_colored).save('/workspace/images/result_thresholded.png')
            return result, has_defect, max_anomaly_score
            
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--url', default='localhost:8001', help='Inference server URL')
    parser.add_argument('--cmap', default='jet', help='Colormap name (e.g., jet, viridis, plasma)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for defect detection')
    args = parser.parse_args()

    start_time = time.time()
    result, has_defect, max_anomaly_score = await infer(args.model, args.image, args.url, args.cmap, args.threshold)
    total_time = time.time() - start_time

    print(f"Processing completed. Check '/workspace/images/result_annotated.png' for the result.")
    print(f"Total processing time: {total_time:.2f} seconds")

if __name__ == '__main__':
    asyncio.run(main())