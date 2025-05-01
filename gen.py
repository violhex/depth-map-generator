import os
import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse
import requests
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers.utils.quantization_config import BitsAndBytesConfig
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

# CPU Throttling
import os
if os.name == 'posix':
    import resource

# -- GPU Controller Class --

class GPUController:
    def __init__(self, target_utilization=50, polling_interval=0.5):
        self.target_utilization = target_utilization
        self.polling_interval = polling_interval
        self.current_util = 0
        self.stop_event = threading.Event()
        self.active = True
        self.adaptive_sleep_time = 0.1
        self.max_sleep_time = 1.0
        self.batch_size = 1
        self.max_batch_size = 4
        self.min_batch_size = 1
        
        # Initialize GPU monitoring
        if GPU_MONITOR_AVAILABLE:
            try:
                self.handle = nvmlDeviceGetHandleByIndex(0)
                self.monitor_thread = threading.Thread(target=self._monitor_gpu, daemon=True)
                self.monitor_thread.start()
                logging.info("GPU monitoring initialized")
            except Exception as e:
                logging.warning(f"Failed to initialize GPU monitoring: {e}")
                self.active = False
        
        # PID control for better throttling
        self.pid_control = {
            'last_error': 0,
            'integral': 0,
            'kp': 0.05,  # Reduced proportional gain for smoother control
            'ki': 0.005, # Reduced integral gain to prevent overshooting
            'kd': 0.02   # Reduced derivative gain for stability
        }
        
        # Add CPU monitoring
        self.cpu_util = 0
        self.cpu_target = target_utilization
        
        # Memory management
        self.last_memory_clear = time.time()
        self.memory_clear_interval = 30  # Clear memory every 30 seconds
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        if GPU_MONITOR_AVAILABLE:
            nvmlShutdown()
            
    def _monitor_gpu(self):
        while not self.stop_event.is_set():
            try:
                # Get GPU utilization
                util = nvmlDeviceGetUtilizationRates(self.handle).gpu
                self.current_util = util
                
                # Get CPU utilization (using psutil)
                import psutil
                self.cpu_util = psutil.cpu_percent(interval=0.1)
                
                # Memory management
                current_time = time.time()
                if current_time - self.last_memory_clear > self.memory_clear_interval:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    self.last_memory_clear = current_time
                
                # PID control for better throttling
                error = max(self.current_util - self.target_utilization, 
                           self.cpu_util - self.cpu_target)
                
                # Calculate PID terms
                p_term = self.pid_control['kp'] * error
                self.pid_control['integral'] = max(0, min(5, 
                                               self.pid_control['integral'] + error))
                i_term = self.pid_control['ki'] * self.pid_control['integral']
                d_term = self.pid_control['kd'] * (error - self.pid_control['last_error'])
                self.pid_control['last_error'] = error
                
                # Apply PID controller output to sleep time
                if error > 0:
                    self.adaptive_sleep_time = min(
                        self.max_sleep_time,
                        max(0.01, p_term + i_term + d_term)
                    )
                else:
                    self.adaptive_sleep_time = max(0.0, self.adaptive_sleep_time * 0.9)
                
                # More conservative batch size control
                if max(util, self.cpu_util) > self.target_utilization + 5:
                    self.batch_size = max(self.min_batch_size, int(self.batch_size * 0.8))
                elif max(util, self.cpu_util) < self.target_utilization - 10:
                    self.batch_size = min(self.max_batch_size, self.batch_size + 1)
                    
                logging.debug(f"GPU: {util}% | CPU: {self.cpu_util}% | Sleep: {self.adaptive_sleep_time:.3f}s | Batch: {self.batch_size}")
                
            except Exception as e:
                logging.warning(f"Monitoring error: {e}")
            time.sleep(self.polling_interval)

    def wait_for_gpu(self):
        if not self.active:
            return
        
        # More conservative waiting policy
        while max(self.current_util, self.cpu_util) > self.target_utilization:
            sleep_time = self.adaptive_sleep_time
            # Add jitter to prevent synchronization issues
            sleep_time *= (0.9 + 0.2 * np.random.random())
            time.sleep(sleep_time)
            
            # Periodically clear memory while waiting
            if time.time() - self.last_memory_clear > self.memory_clear_interval:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                self.last_memory_clear = time.time()

# -- Utility Functions --

def set_resource_limits(cpu_limit_mb=None, nice_value=10):
    """Set CPU limits and process priority"""
    try:
        if cpu_limit_mb:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            os.environ['OMP_NUM_THREADS'] = '2'
            os.environ['MKL_NUM_THREADS'] = '2'
            logging.info(f"CPU thread limit set to 2")
            if os.name == 'posix':
                resource.setrlimit(resource.RLIMIT_AS, (cpu_limit_mb * 1024 * 1024, resource.RLIM_INFINITY))
                logging.info(f"Memory limit set to {cpu_limit_mb}MB")
        if os.name == 'posix':
            os.nice(nice_value)
            logging.info(f"Process priority set to nice={nice_value}")
    except Exception as e:
        logging.warning(f"Failed to set resource limits: {e}")

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

class ProcessingQueue:
    def __init__(self, max_size=5, min_size=1):
        self.queue = queue.Queue()
        self.max_size = max_size
        self.min_size = min_size
        self.current_size = 0
        self.lock = threading.Lock()
        
    def put(self, item):
        with self.lock:
            if self.current_size < self.max_size:
                self.queue.put(item)
                self.current_size += 1
                return True
            return False
            
    def get(self):
        with self.lock:
            if not self.queue.empty():
                self.current_size -= 1
                return self.queue.get()
            return None
            
    def size(self):
        with self.lock:
            return self.current_size
            
    def is_full(self):
        with self.lock:
            return self.current_size >= self.max_size
            
    def is_empty(self):
        with self.lock:
            return self.current_size == 0

def process_and_save(image, filename, output_dir, image_processor, model, device, gpu_controller, resolution_scale=1.0):
    try:
        if resolution_scale != 1.0:
            w, h = image.size
            new_size = (int(w * resolution_scale), int(h * resolution_scale))
            image = image.resize(new_size)
    except Exception as e:
        logging.error(f"Error resizing image {filename}: {e}")
        return

    with torch.no_grad():
        try:
            if gpu_controller:
                gpu_controller.wait_for_gpu()
                
            # Process image and ensure all tensors are on the correct device
            inputs = image_processor(image, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if device.type == 'cuda':
                inputs = {k: v.half() for k, v in inputs.items()}
                
            # Run inference with autocast
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth
                
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Error processing image {filename}: {e}")
            return

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

def process_folder(input_path, output_dir, image_processor, model, device, gpu_controller, resolution_scale=1.0):
    image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    processing_queue = ProcessingQueue(max_size=5)
    
    def worker():
        while True:
            item = processing_queue.get()
            if item is None:
                break
            image, filename = item
            process_and_save(image, filename, output_dir, image_processor, model, device, gpu_controller, resolution_scale)
    
    # Start worker threads
    num_workers = min(4, os.cpu_count() or 1)
    workers = [threading.Thread(target=worker) for _ in range(num_workers)]
    for w in workers:
        w.start()
    
    # Process images
    for filename in image_files:
        full_path = os.path.join(input_path, filename)
        try:
            image = Image.open(full_path).convert("RGB")
            if image.size[0] <= 0 or image.size[1] <= 0:
                logging.warning(f"Invalid image size for {filename}")
                continue
                
            # Wait for queue space
            while processing_queue.is_full():
                time.sleep(0.1)
                
            processing_queue.put((image, filename))
        except Exception as e:
            logging.warning(f"Error opening image {filename}: {e}")
            continue
    
    # Signal workers to stop
    for _ in range(num_workers):
        processing_queue.put(None)
    
    # Wait for workers to finish
    for w in workers:
        w.join()

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

    frame_buffer = []
    frame_idx = 0
    success_count = 0

    with tqdm(total=max_frames, desc="Processing Frames") as pbar:
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_buffer.append((frame_idx, frame))
            frame_idx += 1

            if len(frame_buffer) >= gpu_controller.batch_size if gpu_controller else 1:
                process_frame_batch(frame_buffer, image_processor, model, device, gpu_controller, 
                                  output_dir, out, resolution_scale, width, height)
                success_count += len(frame_buffer)
                frame_buffer = []
                pbar.update(gpu_controller.batch_size if gpu_controller else 1)

    if frame_buffer:
        process_frame_batch(frame_buffer, image_processor, model, device, gpu_controller, 
                          output_dir, out, resolution_scale, width, height)
        success_count += len(frame_buffer)
        pbar.update(len(frame_buffer))

    cap.release()
    out.release()
    duration = time.time() - start_time
    logging.info(f"Successfully processed {success_count}/{max_frames} frames in {duration:.2f} seconds")
    logging.info(f"Saved depth video to: {output_video_path}")

def process_frame_batch(batch, image_processor, model, device, gpu_controller, output_dir, out, resolution_scale, width, height):
    try:
        # Preprocess all frames in batch
        batch_tensors = []
        for idx, frame in batch:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            if resolution_scale != 1.0:
                new_size = (int(width * resolution_scale), int(height * resolution_scale))
                pil_image = pil_image.resize(new_size)
            
            # Process image and ensure all tensors are on the correct device
            with torch.no_grad():
                inputs = image_processor(pil_image, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                if device.type == 'cuda':
                    inputs = {k: v.half() for k, v in inputs.items()}
                batch_tensors.append(inputs)

        # Wait for GPU
        if gpu_controller:
            gpu_controller.wait_for_gpu()

        # Inference
        with torch.no_grad():
            results = []
            for inputs in batch_tensors:
                try:
                    # Ensure all tensors are on the same device
                    model = model.to(device)
                    for k, v in inputs.items():
                        if v.device != device:
                            inputs[k] = v.to(device)
                    
                    # Move model to evaluation mode
                    model.eval()
                    
                    # Run inference
                    with torch.cuda.amp.autocast():
                        outputs = model(**inputs)
                        results.append(outputs.predicted_depth)
                except Exception as e:
                    logging.warning(f"Error during inference: {e}")
                    continue

        # Post-process
        for idx, predicted_depth in enumerate(results):
            prediction = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=(height, width),
                mode="bicubic",
                align_corners=False
            )
            depth = prediction.squeeze().cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth_uint8 = depth.astype(np.uint8)
            save_depth_frame(depth_uint8, out)

    except Exception as e:
        logging.error(f"Batch processing error: {e}")
    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

# -- Main Function --

def set_pytorch_gpu_limits():
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.45)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        logging.info("Set PyTorch to use max 45% of GPU memory")

def set_gpu_power_limit(power_limit_watts):
    try:
        subprocess.run(["nvidia-smi", "-pl", str(power_limit_watts)], check=True)
        logging.info(f"Set GPU power limit to {power_limit_watts}W")
    except Exception as e:
        logging.warning(f"Failed to set GPU power limit: {e}")

def main(input_path, input_type, output_dir, max_frames=None, gpu_throttle=0.0, resolution_scale=1.0, 
         target_gpu_util=50, gpu_power_limit=None, quantized=False):
    # Set low-priority execution
    if os.name == 'posix':
        os.nice(10)
        logging.info("Set process priority to low")
    elif os.name == 'nt':  # Windows
        try:
            import win32api
            import win32process
            import win32con
            pid = win32api.GetCurrentProcessId()
            handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
            win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
            logging.info("Set process priority to below normal")
        except ImportError:
            logging.warning("pywin32 not installed. Process priority control disabled.")
        except Exception as e:
            logging.warning(f"Failed to set process priority: {e}")

    # Set GPU memory limits
    set_pytorch_gpu_limits()
    
    # Set CPU thread limits
    torch.set_num_threads(2)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    
    # Set memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() and not quantized else "cpu")
    logging.info(f"Using device: {device}")

    # Load model with optional quantization
    model_name = "depth-anything/Depth-Anything-V2-Large-hf"
    logging.info(f"Loading model from {model_name}...")

    try:
        # Use fast image processor
        image_processor = AutoImageProcessor.from_pretrained(
            model_name,
            use_fast=True,
            do_resize=True,
            do_rescale=True,
            do_normalize=True
        )

        if quantized:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            model = AutoModelForDepthEstimation.from_pretrained(
                model_name,
                quantization_config=quant_config,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            model = AutoModelForDepthEstimation.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            ).eval()

        # Ensure model is on the correct device and in eval mode
        if device.type == 'cuda':
            model = model.to(device)
            model = model.half()
            model.eval()
            torch.cuda.empty_cache()
            logging.info("GPU cache cleared after model load")

    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    if not os.access(output_dir, os.W_OK):
        logging.error(f"No write access to output directory: {output_dir}")
        return

    if gpu_power_limit and torch.cuda.is_available():
        set_gpu_power_limit(gpu_power_limit)

    # Initialize GPU controller with more conservative settings
    with GPUController(target_utilization=target_gpu_util) as gpu_controller:
        try:
            if input_type == 'folder':
                process_folder(input_path, output_dir, image_processor, model, device, gpu_controller, resolution_scale)
            elif input_type == 'url_file':
                with open(input_path, 'r') as f:
                    urls = [line.strip() for line in f if line.strip()]
                for url in urls:
                    image = download_image(url)
                    if image is None:
                        continue
                    parsed = urlparse(url)
                    filename = os.path.basename(parsed.path) or f"image_{hash(url)}.jpg"
                    process_and_save(image, filename, output_dir, image_processor, model, device, gpu_controller, resolution_scale)
            elif input_type == 'video':
                process_video(input_path, output_dir, image_processor, model, device, gpu_controller, max_frames, resolution_scale)

        except KeyboardInterrupt:
            logging.info("Process interrupted by user")
        except Exception as e:
            logging.error(f"Error during processing: {e}")
        finally:
            # Final cleanup
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
                logging.info("GPU cache cleared after processing")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate depth maps with GPU/CPU usage control')
    parser.add_argument('--input_type', type=str, choices=['folder', 'url_file', 'video'], required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--gpu_throttle', type=float, default=0.0)
    parser.add_argument('--resolution_scale', type=float, default=1.0)
    parser.add_argument('--target_gpu_util', type=int, default=50)
    parser.add_argument('--gpu_power_limit', type=int, default=None)
    parser.add_argument('--quantized', action='store_true', help='Use 4-bit quantized model')
    args = parser.parse_args()

    main(
        input_path=args.input_path,
        input_type=args.input_type,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        gpu_throttle=args.gpu_throttle,
        resolution_scale=args.resolution_scale,
        target_gpu_util=args.target_gpu_util,
        gpu_power_limit=args.gpu_power_limit,
        quantized=args.quantized
    )