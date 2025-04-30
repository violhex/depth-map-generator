import os
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse
import requests
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch.nn.functional as F
import argparse

def download_image(url):
    """Download an image from a URL and return it as a PIL Image."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def save_depth_map(depth_map, output_path):
    """Save the depth map image to the specified path."""
    try:
        depth_map.save(output_path)
    except Exception as e:
        print(f"Error saving depth map to {output_path}: {e}")

def process_and_save(image, filename, output_dir, image_processor, model, device):
    """Process an image and save its depth map."""
    base_name = os.path.splitext(filename)[0]
    output_filename = f"depth_{base_name}.png"
    output_path = os.path.join(output_dir, output_filename)

    try:
        pixel_values = image_processor(image, return_tensors='pt').pixel_values.to(device)
    except Exception as e:
        print(f"Error processing image {filename}: {e}")
        return

    with torch.no_grad():
        outputs = model(pixel_values)
        predicted_depth = outputs.predicted_depth

    original_size = image.size[::-1]  # (width, height) -> (height, width)
    prediction = F.interpolate(
        predicted_depth.unsqueeze(1),
        size=original_size,
        mode="bicubic",
        align_corners=False
    )

    depth = prediction.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth_image = Image.fromarray(depth)

    save_depth_map(depth_image, output_path)

    if os.path.isfile(output_path):
        print(f"Saved depth map: {output_path}")
    else:
        print(f"Failed to save depth map: {output_path}")

def main(input_path, input_type, output_dir):
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    model_name = "depth-anything/Depth-Anything-V2-Large-hf"
    print(f"Loading model from {model_name}...")

    try:
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prepare output directory
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Failed to create output directory: {e}")
            return

    if not os.access(output_dir, os.W_OK):
        print(f"No write access to output directory: {output_dir}")
        return

    # Process input based on type
    if input_type == 'folder':
        print(f"Scanning input folder: {input_path}")
        image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        print(f"Found {len(image_files)} valid image files in {input_path}")

        if not image_files:
            print("No valid image files found. Supported formats: .png, .jpg, .jpeg, .webp")
            return

        for filename in image_files:
            full_path = os.path.join(input_path, filename)
            print(f"Processing image: {filename}")
            try:
                image = Image.open(full_path).convert("RGB")
                if image.size[0] <= 0 or image.size[1] <= 0:
                    print(f"Invalid image size for {filename}")
                    continue
            except Exception as e:
                print(f"Error opening image {filename}: {e}")
                continue

            process_and_save(image, filename, output_dir, image_processor, model, device)

    elif input_type == 'url_file':
        print(f"Reading URL list from: {input_path}")
        try:
            with open(input_path, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            print(f"Found {len(urls)} URLs in file")
        except Exception as e:
            print(f"Error reading URL file: {e}")
            return

        for url in urls:
            print(f"Downloading image from: {url}")
            image = download_image(url)
            if image is None:
                continue
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)
            if not filename:
                filename = f"image_{hash(url)}.jpg"
            process_and_save(image, filename, output_dir, image_processor, model, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate depth maps from images using Hugging Face models.')
    parser.add_argument('--input_type', type=str, choices=['folder', 'url_file'], required=True,
                        help='Input type: folder of images or text file with URLs')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to folder or URL list file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save generated depth maps')

    args = parser.parse_args()

    main(args.input_path, args.input_type, args.output_dir)