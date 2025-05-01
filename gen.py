import os
import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse
import requests
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import time
import logging
import subprocess
import threading
import queue
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GPU Monitoring Support
try:
    from pynvml import (
        nvmlInit, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates, nvmlShutdown
    )
    GPU_MONITOR_AVAILABLE = True
    nvmlInit()
except ImportError:
    logging.warning("pynvml not found. GPU monitoring disabled.")
    GPU_MONITOR_AVAILABLE = False

# -- GPU Controller Class --

class GPUController:
    def __init__(self, target_utilization=50, polling_interval=0.5):
        self.target_utilization = target_utilization
        self.polling_interval = polling_interval
        self.stop_event = threading.Event()
        self.monitor_thread = None
        self.current_util = 100
        self.adaptive_sleep_time = 0.0
        self.max_sleep_time = 2.0
        self.handle = None
        self.active = False

    def start_monitoring(self):
        """Start GPU monitoring thread"""
        if not GPU_MONITOR_AVAILABLE:
            logging.warning("GPU monitoring not available.")
            return
        try:
            self.handle = nvmlDeviceGetHandleByIndex(0)
            self.stop_event.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_gpu, daemon=True)
            self.monitor_thread.start()
            self.active = True
            logging.info(f"GPU monitoring started with target: {self.target_utilization}%")
        except Exception as e:
            logging.warning(f"GPU monitoring failed to start: {e}")

    def stop_monitoring(self):
        """Stop GPU monitoring thread"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_event.set()
            self.monitor_thread.join(timeout=2.0)
            logging.info("GPU monitoring stopped")
        if GPU_MONITOR_AVAILABLE:
            try:
                nvmlShutdown()
            except:
                pass

    def _monitor_gpu(self):
        """Poll GPU utilization and adjust sleep time dynamically"""
        while not self.stop_event.is_set():
            try:
                util = nvmlDeviceGetUtilizationRates(self.handle).gpu
                self.current_util = util
                if util > self.target_utilization:
                    self.adaptive_sleep_time = min(
                        self.max_sleep_time,
                        self.adaptive_sleep_time + 0.05 * (util / self.target_utilization)
                    )
                else:
                    self.adaptive_sleep_time = max(0.0, self.adaptive_sleep_time * 0.9)
                logging.debug(f"GPU Util: {util}% | Sleep: {self.adaptive_sleep_time:.3f}s")
            except Exception as e:
                logging.warning(f"GPU monitoring error: {e}")
            time.sleep(self.polling_interval)

    def wait_for_gpu(self):
        """Wait if GPU utilization exceeds target"""
        if not self.active:
            return
        while self.current_util > self.target_utilization:
            time.sleep(self.adaptive_sleep_time)

    def __enter__(self):
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitoring()

# -- Utility Functions --

def download_image(url):
    try:
        logging.info(f"Downloading image from {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        logging.error(f"Error downloading image from {url}: {e}")
        return None

def save_depth_map(depth_map, output_path):
    try:
        depth_map.save(output_path)
    except Exception as e:
        logging.error(f"Error saving depth map to {output_path}: {e}")

def save_depth_frame(depth_array, output_video_writer):
    try:
        depth_bgr = cv2.cvtColor(depth_array, cv2.COLOR_GRAY2BGR)
        output_video_writer.write(depth_bgr)
    except Exception as e:
        logging.error(f"Error writing depth frame: {e}")

# -- Image Processing --

def process_and_save(image, filename, output_dir, image_processor, model, device, gpu_controller, resolution_scale=1.0):
    try:
        # Resize image if needed
        if resolution_scale != 1.0:
            w, h = image.size
            new_size = (int(w * resolution_scale), int(h * resolution_scale))
            image = image.resize(new_size)

        # Wait for GPU to be under threshold
        if gpu_controller and GPU_MONITOR_AVAILABLE:
            gpu_controller.wait_for_gpu()

        pixel_values = image_processor(image, return_tensors='pt').pixel_values.to(device)
        if device.type == 'cuda':
            pixel_values = {k: v.half() for k, v in pixel_values.items()}
    except Exception as e:
        logging.error(f"Error processing image {filename}: {e}")
        return

    # Inference with GPU throttle
    with torch.no_grad():
        try:
            outputs = model(pixel_values)
            predicted_depth = outputs.predicted_depth
        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.error(f"Out of memory on frame {filename}. Clearing cache.")
                torch.cuda.empty_cache()
                time.sleep(1.0)
            else:
                raise

    original_size = image.size[::-1]
    try:
        prediction = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=original_size,
            mode="bicubic",
            align_corners=False
        )
    except Exception as e:
        logging.warning(f"Error interpolating depth for {filename}: {e}")
        return

    depth = prediction.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth_image = Image.fromarray(depth)

    base_name = os.path.splitext(filename)[0]
    output_filename = f"depth_{base_name}.png"
    output_path = os.path.join(output_dir, output_filename)

    save_depth_map(depth_image, output_path)

    if os.path.isfile(output_path):
        logging.info(f"Saved depth map: {output_path}")
    else:
        logging.warning(f"Failed to save depth map: {output_path}")

# -- Video Processing --

def process_video(video_path, output_dir, image_processor, model, device, gpu_controller, max_frames=None, resolution_scale=1.0):
    if not os.path.isfile(video_path):
        logging.error(f"Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video file: {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count <= 0:
        logging.error(f"Invalid frame count: {frame_count}")
        return

    logging.info(f"Video Info: {width}x{height} @ {fps} FPS, {frame_count} total frames")

    input_video_name = os.path.basename(video_path)
    output_video_name = f"depth_{input_video_name}"
    output_video_path = os.path.join(output_dir, output_video_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        logging.error("Failed to initialize output video writer.")
        return

    max_frames = max_frames or frame_count
    max_frames = min(max_frames, frame_count)

    logging.info(f"Starting depth mapping for up to {max_frames} frames")
    start_time = time.time()

    success_count = 0
    frame_idx = 0

    with tqdm(total=max_frames, desc="Processing Frames") as pbar:
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Failed to read frame {frame_idx}")
                break

            try:
                # Wait for GPU to be under utilization
                if gpu_controller and GPU_MONITOR_AVAILABLE:
                    gpu_controller.wait_for_gpu()

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                if resolution_scale != 1.0:
                    new_size = (int(width * resolution_scale), int(height * resolution_scale))
                    pil_image = pil_image.resize(new_size)

                inputs = image_processor(pil_image, return_tensors='pt').to(device)
                if device.type == 'cuda':
                    inputs = {k: v.half() for k, v in inputs.items()}
                pixel_values = inputs["pixel_values"]
            except Exception as e:
                logging.warning(f"Error pre-processing frame {frame_idx}: {e}")
                continue

            with torch.no_grad():
                try:
                    outputs = model(pixel_values)
                    predicted_depth = outputs.predicted_depth
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logging.warning("Out of memory during inference, clearing cache")
                        torch.cuda.empty_cache()
                        time.sleep(1.0)
                    else:
                        logging.warning(f"Error in model output: {e}")
                    continue

            try:
                prediction = F.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=(height, width),
                    mode="bicubic",
                    align_corners=False
                )
            except Exception as e:
                logging.warning(f"Error interpolating depth for frame {frame_idx}: {e}")
                continue

            try:
                depth = prediction.squeeze().cpu().numpy()
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth_uint8 = depth.astype(np.uint8)

                save_depth_frame(depth_uint8, out)
                success_count += 1
            except Exception as e:
                logging.warning(f"Error saving depth frame {frame_idx}: {e}")
                continue

            if gpu_controller and GPU_MONITOR_AVAILABLE:
                time.sleep(gpu_controller.adaptive_sleep_time)
            elif args.gpu_throttle > 0:
                time.sleep(args.gpu_throttle)

            frame_idx += 1
            pbar.update(1)

            # Clean GPU memory every 10 frames
            if device.type == 'cuda' and frame_idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    cap.release()
    out.release()
    duration = time.time() - start_time
    logging.info(f"Successfully processed {success_count}/{max_frames} frames in {duration:.2f} seconds")
    logging.info(f"Saved depth video to: {output_video_path}")

# -- GPU Setup --

def set_pytorch_gpu_limits():
    """Apply PyTorch-specific GPU limits"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5)  # 50% memory
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        logging.info("Set PyTorch GPU memory to max 50%")
    else:
        logging.info("CUDA not available, skipping GPU limits")

def set_gpu_power_limit(power_limit_watts):
    """System-level power limiting via nvidia-smi"""
    if not torch.cuda.is_available():
        logging.warning("Power limiting requires CUDA GPU")
        return
    try:
        subprocess.run(["nvidia-smi", "-pl", str(power_limit_watts)], check=True)
        logging.info(f"Set GPU power limit to {power_limit_watts}W")
    except Exception as e:
        logging.warning(f"Failed to set GPU power limit: {e}")

# -- Main Function --

def main(input_path, input_type, output_dir, max_frames=None, gpu_throttle=0.0, resolution_scale=1.0, target_gpu_util=50, gpu_power_limit=None):
    # Set GPU memory usage limit
    set_pytorch_gpu_limits()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    model_name = "depth-anything/Depth-Anything-V2-Large-hf"
    logging.info(f"Loading model from {model_name}...")

    try:
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
        model = model.half()  # FP16
        model.eval()
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    if not os.access(output_dir, os.W_OK):
        logging.error(f"No write access to output directory: {output_dir}")
        return

    if gpu_power_limit:
        set_gpu_power_limit(gpu_power_limit)

    # Initialize GPU controller
    gpu_controller = GPUController(target_utilization=target_gpu_util)

    if input_type == 'folder':
        image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        logging.info(f"Found {len(image_files)} valid image files in {input_path}")

        if not image_files:
            logging.warning("No valid image files found. Supported formats: .png, .jpg, .jpeg, .webp")
            return

        for filename in image_files:
            full_path = os.path.join(input_path, filename)
            logging.info(f"Processing image: {filename}")
            try:
                image = Image.open(full_path).convert("RGB")
                if image.size[0] <= 0 or image.size[1] <= 0:
                    logging.warning(f"Invalid image size for {filename}")
                    continue
            except Exception as e:
                logging.warning(f"Error opening image {filename}: {e}")
                continue

            process_and_save(image, filename, output_dir, image_processor, model, device, gpu_controller, resolution_scale)

    elif input_type == 'url_file':
        try:
            with open(input_path, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            logging.info(f"Found {len(urls)} URLs in file")
        except Exception as e:
            logging.error(f"Error reading URL file: {e}")
            return

        for url in urls:
            logging.info(f"Downloading image from: {url}")
            image = download_image(url)
            if image is None:
                continue

            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)
            if not filename:
                filename = f"image_{hash(url)}.jpg"

            process_and_save(image, filename, output_dir, image_processor, model, device, gpu_controller, resolution_scale)

    elif input_type == 'video':
        with gpu_controller:
            process_video(input_path, output_dir, image_processor, model, device, gpu_controller, max_frames, resolution_scale)

    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("GPU cache cleared after processing")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate depth maps from images or videos using Hugging Face models with GPU usage control.')
    parser.add_argument('--input_type', type=str, choices=['folder', 'url_file', 'video'], required=True,
                        help='Input type: folder of images, text file with URLs, or video file')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to folder, URL list file, or video file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save generated depth maps or video')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to process (for testing)')
    parser.add_argument('--gpu_throttle', type=float, default=0.0,
                        help='Seconds to sleep between frames to reduce GPU load (e.g., 0.1)')
    parser.add_argument('--resolution_scale', type=float, default=1.0,
                        help='Scale input resolution (e.g., 0.5 for 50% resolution)')
    parser.add_argument('--target_gpu_util', type=int, default=50,
                        help='Target GPU utilization percentage (default: 50%)')
    parser.add_argument('--gpu_power_limit', type=int, default=None,
                        help='Set GPU power limit in Watts (requires nvidia-smi)')
    args = parser.parse_args()

    main(
        input_path=args.input_path,
        input_type=args.input_type,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        gpu_throttle=args.gpu_throttle,
        resolution_scale=args.resolution_scale,
        target_gpu_util=args.target_gpu_util,
        gpu_power_limit=args.gpu_power_limit
    )