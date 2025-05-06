import os
import torch
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from urllib.parse import urlparse
import requests
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, BitsAndBytesConfig
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import time
import subprocess
import threading
import asyncio
import gc
import aiofiles
import aiohttp
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
from rich.box import ROUNDED

# --- Existing GPUController and Resource Management (largely unchanged, but its usage will change) ---
# (GPUController, set_system_resource_limits, configure_pytorch_memory, set_gpu_power_limit_watts)
# ... (Copy these from your existing gen.py, I will assume they are here for brevity) ...
# Make sure to import psutil in GPUController if it's not already global
# And ensure GPUController's _monitor_resources uses stop_event.wait(timeout) for graceful shutdown

# Set up rich logging
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_time=True, show_level=True, show_path=True)]
)
logger = logging.getLogger("rich")

# --- Constants for Queue Sizes ---
MAX_VIDEO_FRAME_QUEUE_SIZE = 30  # Raw frames from video
MAX_PREPROCESSED_QUEUE_SIZE = 20 # Tensors ready for GPU
MAX_GPU_OUTPUT_QUEUE_SIZE = 20   # Predictions from GPU, ready for postprocessing
MAX_SAVE_QUEUE_SIZE = 30         # Frames ready to be written to disk by saver

# --- Sentinel Value for Queues ---
QUEUE_END_SENTINEL = object()


# --- GPUController (Copied from your existing script, with minor adjustment for asyncio context) ---
# GPU Monitoring Support
GPU_MONITOR_AVAILABLE = False
try:
    from pynvml import (
        nvmlInit, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates, nvmlShutdown, NVMLError
    )
    nvmlInit()
    GPU_MONITOR_AVAILABLE = True
    logger.info(Panel("pynvml initialized successfully. GPU monitoring enabled.", title="[bold green]GPU Monitor", box=ROUNDED))
except ImportError:
    logger.warning(Panel("pynvml not found. GPU monitoring disabled.", title="[yellow]Warning", box=ROUNDED))
except NVMLError as e:
    logger.warning(Panel(f"pynvml initialization failed: {e}. GPU monitoring disabled.", title="[yellow]Warning", box=ROUNDED))
except Exception as e:
    logger.error(Panel(f"Unexpected error during pynvml initialization: {e}", title="[red]Error", box=ROUNDED))


# CPU Throttling & Resource Limits
if os.name == 'posix':
    import resource

class GPUController:
    def __init__(self, user_target_utilization=70, max_allowed_gpu_util=78, polling_interval=0.5,
                 pid_kp=0.05, pid_ki=0.005, pid_kd=0.02, max_batch_size=4, min_batch_size=1):
        self.user_target_utilization = user_target_utilization
        self.max_allowed_gpu_util = max_allowed_gpu_util
        self.pid_target_util = min(user_target_utilization, self.max_allowed_gpu_util - 5)
        self.pid_target_util = max(30, self.pid_target_util)

        self.polling_interval = polling_interval
        self.current_gpu_util = 0
        self.stop_event = threading.Event() # For the monitor thread
        self.active = False
        self.adaptive_sleep_time = 0.05
        self.max_sleep_time = 2.0
        
        self.min_batch_size = max(1, min_batch_size)
        self.max_batch_size = max(self.min_batch_size, max_batch_size)
        self.batch_size = self.min_batch_size
        
        self.handle = None
        if GPU_MONITOR_AVAILABLE:
            try:
                self.handle = nvmlDeviceGetHandleByIndex(0)
                self.active = True
                self.monitor_thread = threading.Thread(target=self._monitor_resources, name="ResourceMonitorThread", daemon=True)
                # No start here, will be started by main async orchestrator
                logger.info(f"GPUController initialized config. PID target: {self.pid_target_util}%, Max: {self.max_allowed_gpu_util}%. Batch: {self.batch_size} (Min:{self.min_batch_size},Max:{self.max_batch_size})")
            except NVMLError as e:
                logger.warning(f"Failed to get GPU handle: {e}. GPUController will be inactive.")
            except Exception as e:
                logger.error(f"Unexpected error getting GPU handle: {e}. GPUController inactive.")
        else:
            logger.info("GPUController is inactive as pynvml is not available.")

        self.pid_control = {'last_error': 0, 'integral': 0, 'kp': pid_kp, 'ki': pid_ki, 'kd': pid_kd}
        self.current_cpu_util = 0
        self.cpu_target = self.pid_target_util
        self.last_memory_clear_time = time.time()
        self.memory_clear_interval = 60

    def start_monitoring(self):
        if self.active and hasattr(self, 'monitor_thread') and not self.monitor_thread.is_alive():
            self.monitor_thread.start()
            logger.info("GPUController monitoring thread started.")

    def stop_monitoring(self):
        self.stop_event.set()
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=self.polling_interval * 3) # Wait a bit longer
            if self.monitor_thread.is_alive():
                 logger.warning("Resource monitor thread did not terminate gracefully.")
        if GPU_MONITOR_AVAILABLE and self.active:
            try:
                nvmlShutdown()
                logger.info("pynvml shutdown.")
            except NVMLError as e:
                logger.warning(f"Error during nvmlShutdown: {e}")

    def _monitor_resources(self):
        import psutil
        pid = os.getpid()
        try:
            py_process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            logger.error(f"Monitor Thread: psutil.Process({pid}) not found. Cannot monitor CPU.")
            return


        logger.info(f"Resource monitor thread ({threading.current_thread().name}) started.")
        while not self.stop_event.is_set():
            try:
                if self.handle:
                    util_rates = nvmlDeviceGetUtilizationRates(self.handle)
                    self.current_gpu_util = util_rates.gpu
                else:
                     self.current_gpu_util = 0

                self.current_cpu_util = py_process.cpu_percent(interval=None) / psutil.cpu_count()

                current_time = time.time()
                if current_time - self.last_memory_clear_time > self.memory_clear_interval:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    self.last_memory_clear_time = current_time
                    logger.debug("Periodic memory clear performed.")
                
                gpu_error = self.current_gpu_util - self.pid_target_util
                cpu_error = self.current_cpu_util - (self.cpu_target * 1.5)
                
                if self.current_gpu_util > self.pid_target_util + 5:
                    effective_error = gpu_error
                else:
                    effective_error = max(gpu_error, cpu_error if cpu_error > 0 else 0)
                
                p_term = self.pid_control['kp'] * effective_error
                dt = self.polling_interval
                self.pid_control['integral'] += effective_error * dt
                integral_max = 20 / (self.pid_control['ki'] + 1e-6)
                self.pid_control['integral'] = max(-integral_max, min(self.pid_control['integral'], integral_max)) 

                if effective_error <= 0:
                    self.pid_control['integral'] *= 0.95

                d_term = self.pid_control['kd'] * (effective_error - self.pid_control['last_error']) / dt
                self.pid_control['last_error'] = effective_error
                
                pid_output = p_term + self.pid_control['ki'] * self.pid_control['integral'] + d_term
                if pid_output > 0:
                    self.adaptive_sleep_time = min(self.max_sleep_time, max(0.01, pid_output))
                else:
                    decay_factor = 0.95 if effective_error > -10 else 0.85
                    self.adaptive_sleep_time = max(0.0, self.adaptive_sleep_time * decay_factor)

                if self.current_gpu_util > self.max_allowed_gpu_util - 3:
                    logger.warning(
                        f"GPU util {self.current_gpu_util:.1f}% critically near limit ({self.max_allowed_gpu_util}%)! "
                        f"Emergency throttle advice: batch to {self.min_batch_size}, sleep to {min(self.max_sleep_time, self.adaptive_sleep_time * 1.5 + 0.1):.3f}s."
                    )
                    # This is advice; the GPU inferencer task will act on it. We still calculate adaptive_sleep_time
                    # And the batch_size is also an advice now.
                    self.adaptive_sleep_time = min(self.max_sleep_time, self.adaptive_sleep_time * 1.5 + 0.1)
                    # Advise minimum batch size
                    advised_batch_size = self.min_batch_size
                else:
                    combined_util = max(self.current_gpu_util, self.current_cpu_util)
                    advised_batch_size = self.batch_size # Start with current
                    if combined_util > self.pid_target_util + 10:
                        if advised_batch_size > self.min_batch_size:
                            advised_batch_size = max(self.min_batch_size, int(advised_batch_size * 0.9))
                    elif combined_util < self.pid_target_util - 15:
                        if self.adaptive_sleep_time < 0.02:
                            if advised_batch_size < self.max_batch_size:
                                advised_batch_size = min(self.max_batch_size, advised_batch_size + 1)
                
                self.batch_size = int(round(advised_batch_size)) # Update the advised batch_size

                logger.debug(
                    f"GPU Advice: Util:{self.current_gpu_util:.1f}%(T:{self.pid_target_util:.1f}%,M:{self.max_allowed_gpu_util:.1f}%)|"
                    f"CPU:{self.current_cpu_util:.1f}%(T:{self.cpu_target:.1f}%)|"
                    f"Sleep:{self.adaptive_sleep_time:.3f}s|Batch:{self.batch_size}|"
                    f"Err:{effective_error:.1f},P:{p_term:.2f},I:{self.pid_control['ki']*self.pid_control['integral']:.2f},D:{d_term:.2f}"
                )
                
            except NVMLError as e:
                logger.warning(f"pynvml error in monitor thread: {e}.")
            except psutil.NoSuchProcess:
                logger.error("Monitor thread: Main process no longer exists. Shutting down monitor.")
                break # Exit the loop
            except Exception as e:
                logger.error(f"Unexpected error in _monitor_resources: {e}", exc_info=True)
            
            # Use event wait with timeout for cleaner exit
            if self.stop_event.wait(timeout=self.polling_interval):
                break # Stop event was set
        logger.info(f"Resource monitor thread ({threading.current_thread().name}) finished.")


    def get_current_adaptive_sleep_time(self):
        return self.adaptive_sleep_time if self.active else 0.01

    def get_advised_batch_size(self):
        return self.batch_size if self.active else 1

    def is_gpu_overloaded(self):
        if not self.active:
            return False
        return self.current_gpu_util > self.max_allowed_gpu_util

# Utility Functions (set_system_resource_limits, download_image (sync), save_depth_map, save_depth_frame (sync))
# configure_pytorch_memory, set_gpu_power_limit_watts
# Copied from your existing script for brevity. download_image will be replaced by an async version.
def set_system_resource_limits(nice_value=10):
    try:
        num_cores = os.cpu_count()
        constrained_threads = max(1, min(num_cores // 2 if num_cores else 1, 4)) 
        torch.set_num_threads(constrained_threads)
        os.environ['OMP_NUM_THREADS'] = str(constrained_threads)
        os.environ['MKL_NUM_THREADS'] = str(constrained_threads)
        logger.info(f"Global thread limit set to {constrained_threads} for PyTorch/OMP/MKL.")
        if os.name == 'posix':
            os.nice(nice_value)
            logger.info(f"Process priority set to nice={nice_value}")
        elif os.name == 'nt': 
            import win32api, win32process, win32con
            pid = win32api.GetCurrentProcessId()
            handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
            win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
            logger.info("Process priority set to Below Normal (Windows).")
    except ImportError:
        logger.warning("win32api not found for setting process priority on Windows.")
    except Exception as e:
        logger.warning(f"Failed to set some resource limits: {e}")

async def async_download_image(session, url, pbar=None):
    try:
        logger.debug(f"Downloading image from {url}")
        async with session.get(url, timeout=20) as response:
            response.raise_for_status()
            content_type = response.headers.get('content-type')
            if content_type and not content_type.lower().startswith('image/'):
                 logger.warning(f"URL {url} returned non-image content type: {content_type}")
            
            image_bytes = await response.read()
            # To avoid blocking the event loop with Image.open, run in executor if it's heavy
            loop = asyncio.get_running_loop()
            img = await loop.run_in_executor(None, lambda: Image.open(BytesIO(image_bytes)).convert("RGB"))
            logger.info(f"Successfully downloaded image from {url} ({img.width}x{img.height})")
            if pbar: pbar.update(1)
            return img
    except (aiohttp.ClientError, asyncio.TimeoutError) as e: # More specific aiohttp errors
        logger.error(f"Network error downloading {url}: {e}")
    except UnidentifiedImageError:
         logger.error(f"Cannot identify image file from {url}.")
    except Exception as e:
        logger.error(f"Error processing downloaded image from {url}: {e}", exc_info=True)
    if pbar: pbar.update(1) # Ensure pbar updates even on failure for this URL
    return None

def save_depth_map_sync(depth_map_image, output_path): # Renamed to avoid conflict if needed
    try:
        depth_map_image.save(output_path)
        logger.info(f"Saved depth map: {output_path}")
    except Exception as e:
        logger.error(f"Error saving depth map to {output_path}: {e}")

def save_depth_frame_sync(depth_array_uint8, output_video_writer):
    try:
        if len(depth_array_uint8.shape) == 3 and depth_array_uint8.shape[2] == 3:
             depth_bgr = depth_array_uint8
        elif len(depth_array_uint8.shape) == 2:
             depth_bgr = cv2.cvtColor(depth_array_uint8, cv2.COLOR_GRAY2BGR)
        else:
             logger.error(f"Invalid shape for depth frame: {depth_array_uint8.shape}")
             return
        output_video_writer.write(depth_bgr)
    except Exception as e:
        logger.error(f"Error writing depth frame: {e}")

def configure_pytorch_memory(max_split_mb=128, memory_fraction=0.75): 
    if torch.cuda.is_available():
        try:
            conf_val = f'max_split_size_mb:{max_split_mb}'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = conf_val
            logger.info(f"PYTORCH_CUDA_ALLOC_CONF set to '{conf_val}'.")
            if 0.0 < memory_fraction <= 1.0:
                current_device = torch.cuda.current_device() 
                torch.cuda.set_per_process_memory_fraction(memory_fraction, device=current_device)
                logger.info(f"PyTorch GPU memory usage per process limited to {memory_fraction*100:.0f}% on device {current_device}.")
            else:
                logger.warning(f"Invalid memory_fraction {memory_fraction}. Not setting limit.")
        except RuntimeError as e:
            if "driver shutting down" in str(e).lower(): logger.error(f"CUDA runtime error (driver shutting down?) while setting memory fraction: {e}")
            else: logger.warning(f"Failed to configure PyTorch GPU memory settings: {e}")
        except Exception as e: logger.warning(f"An unexpected error occurred while configuring PyTorch memory: {e}")

def set_gpu_power_limit_watts(power_limit_watts):
    if not GPU_MONITOR_AVAILABLE:
        logger.warning("Cannot set GPU power limit: pynvml not available/active.")
        return
    try:
        command = ["nvidia-smi", "-pl", str(power_limit_watts)]
        logger.info(f"Attempting to run: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        logger.info(f"Successfully set GPU power limit to {power_limit_watts}W. Output: {result.stdout.strip()}")
    except FileNotFoundError: logger.warning("nvidia-smi command not found.")
    except subprocess.TimeoutExpired: logger.warning("nvidia-smi command timed out.")
    except subprocess.CalledProcessError as e: logger.warning(f"Failed to set GPU power limit via nvidia-smi (admin privileges?): {e.stderr.strip()}.")
    except Exception as e: logger.error(f"Unexpected error setting GPU power limit: {e}", exc_info=True)


# --- Asynchronous Pipeline Components ---

async def video_frame_feeder(video_path, frame_queue: asyncio.Queue, max_frames_to_process, pbar):
    """Reads frames from video and puts them onto a queue."""
    task_name = asyncio.current_task().get_name()
    logger.info(f"[{task_name}] Starting video frame feeder for {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"[{task_name}] Failed to open video: {video_path}")
        await frame_queue.put(QUEUE_END_SENTINEL)
        return

    frames_read = 0
    try:
        while True:
            if max_frames_to_process is not None and frames_read >= max_frames_to_process:
                logger.info(f"[{task_name}] Reached max_frames_to_process ({max_frames_to_process}).")
                break
            
            # Run blocking cv2.read in executor to not block event loop
            loop = asyncio.get_running_loop()
            ret, frame = await loop.run_in_executor(None, cap.read)

            if not ret:
                logger.info(f"[{task_name}] End of video or cannot read frame.")
                break
            
            await frame_queue.put({"id": frames_read, "data": frame, "type": "video_frame"})
            pbar.update(1) # Update for frame read
            frames_read += 1
            await asyncio.sleep(0.001) # Tiny yield to allow other tasks
    except Exception as e:
        logger.error(f"[{task_name}] Error in video_frame_feeder: {e}", exc_info=True)
    finally:
        cap.release()
        await frame_queue.put(QUEUE_END_SENTINEL)
        logger.info(f"[{task_name}] Video frame feeder finished. Read {frames_read} frames.")


async def image_path_feeder(folder_path, item_queue: asyncio.Queue, pbar):
    task_name = asyncio.current_task().get_name()
    logger.info(f"[{task_name}] Starting image path feeder for folder: {folder_path}")
    try:
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'))]
        pbar.reset(total=len(image_files))
        for filename in image_files:
            full_path = os.path.join(folder_path, filename)
            await item_queue.put({"id": filename, "data": full_path, "type": "image_path"})
            pbar.update(1)
            await asyncio.sleep(0.001)
    except Exception as e:
        logger.error(f"[{task_name}] Error in image_path_feeder: {e}", exc_info=True)
    finally:
        await item_queue.put(QUEUE_END_SENTINEL)
        logger.info(f"[{task_name}] Image path feeder finished.")


async def url_feeder(url_file_path, item_queue: asyncio.Queue, pbar):
    task_name = asyncio.current_task().get_name()
    logger.info(f"[{task_name}] Starting URL feeder for file: {url_file_path}")
    try:
        async with aiofiles.open(url_file_path, 'r') as f:
            urls = [line.strip() for line in await f.readlines() if line.strip() and not line.startswith('#')]
        pbar.reset(total=len(urls)) # pbar here counts URLs to be submitted for download
        for i, url_str in enumerate(urls):
            await item_queue.put({"id": f"url_{i}", "data": url_str, "type": "image_url"})
            # pbar update will happen in the downloader
            await asyncio.sleep(0.001)
    except FileNotFoundError:
        logger.error(f"[{task_name}] URL file not found: {url_file_path}")
    except Exception as e:
        logger.error(f"[{task_name}] Error reading URL file {url_file_path}: {e}", exc_info=True)
    finally:
        await item_queue.put(QUEUE_END_SENTINEL)
        logger.info(f"[{task_name}] URL feeder finished.")


async def preprocessor_worker(
    worker_id: int,
    input_queue: asyncio.Queue,      # Receives items from feeder (frames, paths, urls)
    preprocessed_queue: asyncio.Queue, # Sends data ready for GPU
    image_processor_hf,             # HuggingFace image processor
    resolution_scale: float,
    pbar_preprocess,                # TQDM for preprocessing
    download_session: aiohttp.ClientSession = None # For URL input type
):
    """Worker that loads/downloads image, resizes, and uses HF image_processor."""
    task_name = asyncio.current_task().get_name()
    logger.info(f"[{task_name}] Preprocessor Worker-{worker_id} started.")
    loop = asyncio.get_running_loop()

    while True:
        try:
            item = await input_queue.get()
            if item is QUEUE_END_SENTINEL:
                await preprocessed_queue.put(QUEUE_END_SENTINEL) # Propagate sentinel
                input_queue.task_done() # Acknowledge sentinel
                logger.info(f"[{task_name}] Preprocessor Worker-{worker_id} received sentinel, shutting down.")
                break

            task_id = item["id"]
            item_type = item["type"]
            item_data = item["data"]
            pil_image = None
            original_size = None

            if item_type == "video_frame":
                # item_data is a numpy BGR frame
                rgb_frame = cv2.cvtColor(item_data, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
            elif item_type == "image_path":
                # item_data is a file path
                try:
                    # Run blocking Image.open in executor
                    pil_image = await loop.run_in_executor(None, lambda: Image.open(item_data).convert("RGB"))
                except FileNotFoundError:
                    logger.warning(f"[{task_name}] Image not found: {item_data}")
                    input_queue.task_done(); pbar_preprocess.update(1); continue
                except UnidentifiedImageError:
                    logger.warning(f"[{task_name}] Cannot identify image: {item_data}")
                    input_queue.task_done(); pbar_preprocess.update(1); continue
            elif item_type == "image_url":
                if download_session:
                    pil_image = await async_download_image(download_session, item_data)
                else:
                    logger.warning(f"[{task_name}] Download session not provided for URL: {item_data}")
                    input_queue.task_done(); pbar_preprocess.update(1); continue
            
            if pil_image is None:
                input_queue.task_done(); pbar_preprocess.update(1); continue

            original_size = pil_image.size # W, H

            # Preprocessing: Resizing (if needed)
            target_pil_image = pil_image
            if resolution_scale != 1.0:
                try:
                    w, h = pil_image.size
                    new_size = (int(w * resolution_scale), int(h * resolution_scale))
                    if new_size[0] > 0 and new_size[1] > 0:
                        # Run blocking resize in executor
                        target_pil_image = await loop.run_in_executor(None, lambda: pil_image.resize(new_size, Image.Resampling.LANCZOS))
                    else:
                        logger.warning(f"[{task_name}] Invalid resize for {task_id} ({new_size}), skipping.")
                        input_queue.task_done(); pbar_preprocess.update(1); continue
                except Exception as e:
                    logger.error(f"[{task_name}] Error resizing image {task_id}: {e}")
                    input_queue.task_done(); pbar_preprocess.update(1); continue
            
            del pil_image # Free original if it was copied

            # HuggingFace image_processor (usually CPU-bound, can be slow)
            # Run in executor to prevent blocking the event loop
            inputs_cpu = await loop.run_in_executor(None, lambda: image_processor_hf(images=target_pil_image, return_tensors='pt'))
            del target_pil_image

            await preprocessed_queue.put({
                "id": task_id,
                "inputs_cpu": inputs_cpu,
                "original_size": original_size, # For final interpolation
                "type": item_type # Pass along type for saver
            })
            pbar_preprocess.update(1)
            input_queue.task_done()

        except asyncio.CancelledError:
            logger.info(f"[{task_name}] Preprocessor Worker-{worker_id} cancelled.")
            break
        except Exception as e:
            logger.error(f"[{task_name}] Error in Preprocessor Worker-{worker_id}: {e}", exc_info=True)
            # If an error occurs, ensure task_done is called if item was retrieved
            if 'item' in locals() and item is not QUEUE_END_SENTINEL :
                input_queue.task_done()
            pbar_preprocess.update(1) # Assume one item processed/failed


async def gpu_inferencer_task(
    preprocessed_queue: asyncio.Queue,
    gpu_output_queue: asyncio.Queue,
    model, device, gpu_controller: GPUController,
    pbar_gpu_infer # TQDM for GPU inference
):
    """Dedicated task for running model inference on the GPU."""
    task_name = asyncio.current_task().get_name()
    logger.info(f"[{task_name}] GPU Inferencer started, using device: {device}")
    
    batch_buffer = []
    last_batch_time = time.monotonic()

    while True:
        try:
            current_advised_batch_size = gpu_controller.get_advised_batch_size()
            item = None # Define item here for wider scope

            # Try to get an item, with a timeout to allow checking batch conditions
            try:
                # Timeout allows checking if buffer should be processed even if queue is slow
                timeout_val = 0.05 if batch_buffer else None # Short timeout if buffer has items
                item = await asyncio.wait_for(preprocessed_queue.get(), timeout=timeout_val)
            except asyncio.TimeoutError:
                # If timeout and buffer has items, process the buffer
                if batch_buffer:
                    pass # Proceed to process current batch_buffer
                else:
                    continue # No items in buffer, and queue was empty during timeout

            if item: # If an item was fetched
                if item is QUEUE_END_SENTINEL:
                    if batch_buffer: # Process any remaining items in buffer
                        logger.info(f"[{task_name}] GPU Inferencer received sentinel, processing remaining {len(batch_buffer)} items.")
                        # Fall through to process batch_buffer
                    else:
                        await gpu_output_queue.put(QUEUE_END_SENTINEL) # Propagate sentinel
                        preprocessed_queue.task_done() # Acknowledge sentinel from input queue
                        logger.info(f"[{task_name}] GPU Inferencer received sentinel and buffer empty, shutting down.")
                        break
                else:
                    batch_buffer.append(item)

            # Check if batch is full or conditions met to process
            # Conditions: batch full OR (sentinel received and buffer not empty) OR (timeout occurred and buffer not empty)
            should_process_batch = (
                len(batch_buffer) >= current_advised_batch_size or
                (item is QUEUE_END_SENTINEL and batch_buffer) or
                (item is None and batch_buffer and (time.monotonic() - last_batch_time > 0.2)) # Process if items waiting > 0.2s
            )

            if not batch_buffer or not should_process_batch:
                if item and item is not QUEUE_END_SENTINEL : preprocessed_queue.task_done() # Ack if item processed individually
                continue


            # --- Perform GPU Inference on the Batch ---
            actual_batch_size = len(batch_buffer)
            logger.debug(f"[{task_name}] Processing batch of size {actual_batch_size} (advised: {current_advised_batch_size})")
            
            # Collate inputs
            batch_ids = [b_item["id"] for b_item in batch_buffer]
            batch_inputs_cpu_list = [b_item["inputs_cpu"] for b_item in batch_buffer]
            batch_original_sizes = [b_item["original_size"] for b_item in batch_buffer]
            batch_types = [b_item["type"] for b_item in batch_buffer]

            # This assumes inputs_cpu is a dict of tensors. Need proper batching.
            # For HuggingFace, processor usually handles list of images. Here we have list of processed dicts.
            # Simplistic batching: stack if possible, or process one-by-one if shapes differ (HF processor should ensure same shapes if from list)
            # For now, assuming HF processor already batched if multiple images were passed to it.
            # If preprocessor sends one-by-one, then inputs_cpu is for a single image.
            # The following code assumes `inputs_cpu` from preprocessor is for a single image.
            # We need to collate these single-image `inputs_cpu` dicts into a batch of dicts.
            
            collated_inputs_cpu = {}
            if batch_inputs_cpu_list:
                # Assuming all dicts have same keys
                for key in batch_inputs_cpu_list[0].keys():
                    collated_inputs_cpu[key] = torch.cat([d[key] for d in batch_inputs_cpu_list], dim=0)

            inputs_gpu = {k: v.to(device, non_blocking=True) for k, v in collated_inputs_cpu.items()}
            del collated_inputs_cpu, batch_inputs_cpu_list

            # Advised sleep from GPUController before heavy compute
            if gpu_controller.is_gpu_overloaded():
                 sleep_time = gpu_controller.get_current_adaptive_sleep_time() + 0.1 # Add fixed penalty
                 logger.info(f"[{task_name}] GPU overloaded ({gpu_controller.current_gpu_util:.1f}%). Inferencer sleeping for {sleep_time:.3f}s before batch.")
                 await asyncio.sleep(sleep_time)
            
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(**inputs_gpu)
                predicted_depths_gpu_batch = outputs.predicted_depth
            
            del inputs_gpu
            if hasattr(outputs, 'hidden_states'): del outputs.hidden_states
            if hasattr(outputs, 'attentions'): del outputs.attentions
            del outputs
            # No torch.cuda.synchronize() here unless strictly needed, let it be async

            # Send individual results to output queue
            for i in range(actual_batch_size):
                await gpu_output_queue.put({
                    "id": batch_ids[i],
                    "predicted_depth_gpu": predicted_depths_gpu_batch[i].unsqueeze(0), # Keep batch dim for interpolate
                    "original_size": batch_original_sizes[i],
                    "type": batch_types[i]
                })
            
            pbar_gpu_infer.update(actual_batch_size)
            for _ in range(actual_batch_size): preprocessed_queue.task_done() # Ack each item from input queue

            batch_buffer = [] # Clear buffer
            last_batch_time = time.monotonic()

            # Short sleep after a batch, potentially adjusted by GPUController
            # This is the primary mechanism for this task to control its rate
            sleep_duration = gpu_controller.get_current_adaptive_sleep_time()
            if sleep_duration > 0.001: # Only sleep if value is meaningful
                logger.debug(f"[{task_name}] GPU Inferencer sleeping for {sleep_duration:.3f}s after batch.")
                await asyncio.sleep(sleep_duration)
            else:
                await asyncio.sleep(0) # Yield control quickly

            if device.type == 'cuda': torch.cuda.empty_cache() # More frequent cache clearing

        except asyncio.CancelledError:
            logger.info(f"[{task_name}] GPU Inferencer task cancelled.")
            # Clean up tasks from queue if cancelled mid-batch
            for b_item in batch_buffer: preprocessed_queue.task_done()
            break
        except Exception as e:
            logger.error(f"[{task_name}] Error in GPU Inferencer: {e}", exc_info=True)
            if 'item' in locals() and item is not QUEUE_END_SENTINEL: # Check if item was defined
                 for b_item in batch_buffer: preprocessed_queue.task_done() # Ack all in buffer
            batch_buffer = [] # Clear buffer on error
            await asyncio.sleep(1) # Sleep a bit after an error


async def postprocessor_saver_worker(
    worker_id: int,
    gpu_output_queue: asyncio.Queue,
    # save_queue: asyncio.Queue, # If saving is also parallelized and slow
    output_dir: str,
    device_for_cpu_transfer: str, # 'cpu'
    pbar_postprocess, # TQDM for post-processing
    video_writer_dict: dict, # To hold video writers if type is video
    video_fps_dict: dict # To hold fps for video
):
    """Worker that takes GPU output, interpolates, normalizes, and prepares for saving."""
    task_name = asyncio.current_task().get_name()
    logger.info(f"[{task_name}] Postprocessor Worker-{worker_id} started.")
    loop = asyncio.get_running_loop()

    while True:
        try:
            item = await gpu_output_queue.get()
            if item is QUEUE_END_SENTINEL:
                # await save_queue.put(QUEUE_END_SENTINEL) # Propagate if there's a save_queue
                gpu_output_queue.task_done()
                logger.info(f"[{task_name}] Postprocessor Worker-{worker_id} received sentinel, shutting down.")
                break
            
            task_id = item["id"]
            predicted_depth_gpu = item["predicted_depth_gpu"]
            original_size = item["original_size"] # W, H
            item_type = item["type"]

            # Interpolation (GPU)
            target_interpolation_size = (original_size[1], original_size[0]) # H, W for F.interpolate
            with torch.no_grad(): # Ensure no_grad for ops not needing gradients
                prediction_gpu = F.interpolate(
                    predicted_depth_gpu.unsqueeze(1) if len(predicted_depth_gpu.shape) == 3 else predicted_depth_gpu, # Ensure 4D [N,C,H,W]
                    size=target_interpolation_size,
                    mode="bicubic",
                    align_corners=False
                )
            del predicted_depth_gpu # Free GPU tensor

            # Move to CPU and Normalize (CPU-bound)
            # Run in executor if numpy operations are heavy
            def _normalize_on_cpu(pred_gpu_tensor):
                depth_cpu_numpy = pred_gpu_tensor.squeeze().to(torch.float32).cpu().numpy()
                # del pred_gpu_tensor # This tensor is local to this function
                
                depth_min = depth_cpu_numpy.min()
                depth_max = depth_cpu_numpy.max()
                if depth_max - depth_min > 1e-6:
                    depth_normalized = (depth_cpu_numpy - depth_min) / (depth_max - depth_min) * 255.0
                else:
                    depth_normalized = np.zeros_like(depth_cpu_numpy)
                return depth_normalized.astype(np.uint8)

            depth_uint8_numpy = await loop.run_in_executor(None, _normalize_on_cpu, prediction_gpu)
            del prediction_gpu # Free the tensor after passing to executor
            if torch.cuda.is_available(): torch.cuda.empty_cache()


            # --- Saving Logic ---
            if item_type == "video_frame":
                video_id = "default_video" # Assuming single video for now, extend if multiple
                if video_id not in video_writer_dict or video_writer_dict[video_id] is None:
                    # This setup should ideally happen once in the main orchestrator
                    logger.error(f"[{task_name}] Video writer for '{video_id}' not initialized for frame {task_id}.")
                else:
                    # cv2.VideoWriter.write is blocking
                    await loop.run_in_executor(None, save_depth_frame_sync, depth_uint8_numpy, video_writer_dict[video_id])
            
            elif item_type in ["image_path", "image_url"]:
                depth_image_pil = Image.fromarray(depth_uint8_numpy)
                base_name = os.path.splitext(str(task_id))[0] if isinstance(task_id, str) else f"image_{task_id}_{time.time_ns()}"
                output_filename = f"depth_{base_name}.png"
                output_path = os.path.join(output_dir, output_filename)
                # Image.save can be blocking
                await loop.run_in_executor(None, save_depth_map_sync, depth_image_pil, output_path)
                del depth_image_pil

            del depth_uint8_numpy
            pbar_postprocess.update(1)
            gpu_output_queue.task_done()
            gc.collect()

        except asyncio.CancelledError:
            logger.info(f"[{task_name}] Postprocessor Worker-{worker_id} cancelled.")
            break
        except Exception as e:
            logger.error(f"[{task_name}] Error in Postprocessor Worker-{worker_id}: {e}", exc_info=True)
            if 'item' in locals() and item is not QUEUE_END_SENTINEL: gpu_output_queue.task_done()
            pbar_postprocess.update(1)


async def run_pipeline(args):
    """Orchestrates the entire asynchronous pipeline."""
    start_time_main = time.time()
    
    # --- Global Setup ---
    set_system_resource_limits()
    configure_pytorch_memory(memory_fraction=args.memory_fraction)
    if args.gpu_power_limit and torch.cuda.is_available():
        set_gpu_power_limit_watts(args.gpu_power_limit)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    # --- Model Loading ---
    model_name_hf = args.model_name if args.model_name else "depth-anything/Depth-Anything-V2-Large-hf"
    logger.info(f"Loading model: {model_name_hf} (Quantized: {args.quantized})")
    model = None
    image_processor_hf = None # Renamed to avoid conflict
    try:
        image_processor_hf = AutoImageProcessor.from_pretrained(model_name_hf)
        model_load_kwargs = {"low_cpu_mem_usage": True}
        if args.quantized:
            if device.type == 'cpu':
                logger.warning("BitsAndBytes 4-bit quantization is CUDA-only. Loading standard FP32 model on CPU.")
                model = AutoModelForDepthEstimation.from_pretrained(model_name_hf, **model_load_kwargs)
            else:
                quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
                model_load_kwargs["quantization_config"] = quant_config
                model_load_kwargs["device_map"] = "auto"
                model = AutoModelForDepthEstimation.from_pretrained(model_name_hf, **model_load_kwargs)
                logger.info("Loaded 4-bit quantized model with device_map='auto'.")
        else:
            if device.type == 'cuda':
                model_load_kwargs["torch_dtype"] = torch.float16
                model_load_kwargs["device_map"] = "auto"
                model = AutoModelForDepthEstimation.from_pretrained(model_name_hf, **model_load_kwargs)
                logger.info("Loaded FP16 model with device_map='auto' for CUDA.")
            else:
                 model = AutoModelForDepthEstimation.from_pretrained(model_name_hf, **model_load_kwargs)
                 model = model.to(device)
                 logger.info("Loaded FP32 model on CPU.")
        model.eval()
        if device.type == 'cuda': torch.cuda.empty_cache(); gc.collect()
        logger.info("Model loaded and GPU cache cleared.")
    except Exception as e:
        logger.error(f"Fatal error loading model or image processor: {e}", exc_info=True)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Initialize Queues ---
    # Feeder queue can be smaller as it directly feeds preprocessors
    feeder_item_queue = asyncio.Queue(maxsize=MAX_PREPROCESSED_QUEUE_SIZE * 2) # Items from file/video reader
    preprocessed_item_queue = asyncio.Queue(maxsize=MAX_PREPROCESSED_QUEUE_SIZE)
    gpu_output_item_queue = asyncio.Queue(maxsize=MAX_GPU_OUTPUT_QUEUE_SIZE)
    # save_item_queue = asyncio.Queue(maxsize=MAX_SAVE_QUEUE_SIZE) # If saving is a separate stage

    # --- Initialize GPUController ---
    gpu_controller = GPUController(
        user_target_utilization=args.target_gpu_util,
        max_allowed_gpu_util=args.gpu_max_allowed_util,
        min_batch_size=args.batch_size_min,
        max_batch_size=args.batch_size_max
    )
    gpu_controller.start_monitoring() # Start the thread

    # --- Prepare for specific input type ---
    feeder_task = None
    total_items_expected = 0
    video_writer_dict = {} # For video processing
    video_fps_dict = {}    # For video processing
    original_video_path_for_audio = None # For audio remuxing

    # Initialize TQDMs
    # pbar_feed = tqdm(desc="Feeder", unit="item", position=0, dynamic_ncols=True) # Total updated later
    pbar_preprocess = tqdm(desc="PreProcessing", unit="item", position=1, dynamic_ncols=True)
    pbar_gpu_infer = tqdm(desc="GPU Inference", unit="item", position=2, dynamic_ncols=True)
    pbar_postprocess = tqdm(desc="PostProcessing/Saving", unit="item", position=3, dynamic_ncols=True)


    if args.input_type == 'video':
        cap_temp = cv2.VideoCapture(args.input_path)
        total_items_expected = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.max_frames is not None:
            total_items_expected = min(total_items_expected, args.max_frames)
        
        video_fps = cap_temp.get(cv2.CAP_PROP_FPS)
        vid_w = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_temp.release()

        video_id = "default_video" # Key for dicts
        video_fps_dict[video_id] = video_fps
        output_video_basename = f"depth_{os.path.splitext(os.path.basename(args.input_path))[0]}.mp4"
        depth_video_only_path = os.path.join(args.output_dir, output_video_basename)
        final_output_video_path = os.path.join(args.output_dir, f"final_{output_video_basename}")
        original_video_path_for_audio = args.input_path


        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer_dict[video_id] = cv2.VideoWriter(depth_video_only_path, fourcc, video_fps, (vid_w, vid_h))
        if not video_writer_dict[video_id].isOpened():
            logger.error(f"Failed to open video writer for {depth_video_only_path}")
            gpu_controller.stop_monitoring(); return

        # pbar_feed.reset(total=total_items_expected)
        pbar_feed = tqdm(desc="Video Frames Read", unit="frame", total=total_items_expected, position=0, dynamic_ncols=True)
        feeder_task = asyncio.create_task(video_frame_feeder(args.input_path, feeder_item_queue, args.max_frames, pbar_feed), name="VideoFrameFeeder")
    
    elif args.input_type == 'folder':
        # Count files for TQDM total
        image_files_temp = [f for f in os.listdir(args.input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'))]
        total_items_expected = len(image_files_temp)
        # pbar_feed.reset(total=total_items_expected)
        pbar_feed = tqdm(desc="Image Files Queued", unit="file", total=total_items_expected, position=0, dynamic_ncols=True)
        feeder_task = asyncio.create_task(image_path_feeder(args.input_path, feeder_item_queue, pbar_feed), name="ImagePathFeeder")

    elif args.input_type == 'url_file':
        try:
            with open(args.input_path, 'r') as f_urls:
                urls_temp = [line.strip() for line in f_urls if line.strip() and not line.startswith('#')]
            total_items_expected = len(urls_temp)
        except FileNotFoundError:
            logger.error(f"URL file not found for counting: {args.input_path}")
            gpu_controller.stop_monitoring(); return
        # pbar_feed.reset(total=total_items_expected) # For URLs, pbar_feed is for URLs submitted for download.
        pbar_feed = tqdm(desc="URLs Submitted", unit="url", total=total_items_expected, position=0, dynamic_ncols=True)
        feeder_task = asyncio.create_task(url_feeder(args.input_path, feeder_item_queue, pbar_feed), name="URLFeeder")
    
    if total_items_expected > 0:
        pbar_preprocess.reset(total=total_items_expected)
        pbar_gpu_infer.reset(total=total_items_expected)
        pbar_postprocess.reset(total=total_items_expected)


    # --- Create Worker Tasks ---
    num_cpu_workers = max(1, min(os.cpu_count() // 2 if os.cpu_count() else 1, 4))
    logger.info(f"Starting {num_cpu_workers} preprocessor and postprocessor workers.")

    download_session = None
    if args.input_type == 'url_file':
        download_session = aiohttp.ClientSession(headers={'User-Agent': 'Mozilla/5.0'})


    preprocessor_tasks = [
        asyncio.create_task(preprocessor_worker(
            i, feeder_item_queue, preprocessed_item_queue, image_processor_hf, args.resolution_scale, pbar_preprocess, download_session
        ), name=f"Preprocessor-{i}") for i in range(num_cpu_workers)
    ]

    gpu_infer_task = asyncio.create_task(gpu_inferencer_task(
        preprocessed_item_queue, gpu_output_item_queue, model, device, gpu_controller, pbar_gpu_infer
    ), name="GPUInferencer")

    postprocessor_tasks = [
        asyncio.create_task(postprocessor_saver_worker(
            i, gpu_output_item_queue, args.output_dir, 'cpu', pbar_postprocess, video_writer_dict, video_fps_dict
        ), name=f"Postprocessor-{i}") for i in range(num_cpu_workers) # Can adjust num_cpu_workers for saving
    ]

    all_processing_tasks = preprocessor_tasks + [gpu_infer_task] + postprocessor_tasks

    # --- Wait for Feeder and then Propagated Sentinels ---
    try:
        if feeder_task:
            await feeder_task 
            logger.info("Feeder task completed.")
        
        # Wait for all items to flow through the queues and be processed
        await feeder_item_queue.join()
        logger.info("Feeder item queue joined (all items taken by preprocessors).")
        await preprocessed_item_queue.join()
        logger.info("Preprocessed item queue joined (all items taken by GPU inferencer).")
        await gpu_output_item_queue.join()
        logger.info("GPU output item queue joined (all items taken by postprocessors).")

        logger.info("All queues joined. All primary processing should be complete.")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received by orchestrator. Cancelling tasks...")
        if feeder_task and not feeder_task.done(): feeder_task.cancel()
        for task in all_processing_tasks:
            if not task.done(): task.cancel()
    except Exception as e:
        logger.error(f"Error in main pipeline orchestration: {e}", exc_info=True)
        if feeder_task and not feeder_task.done(): feeder_task.cancel()
        for task in all_processing_tasks:
            if not task.done(): task.cancel()
    finally:
        # --- Cleanup ---
        logger.info("Initiating final cleanup of pipeline resources.")
        
        # Ensure all worker tasks are awaited to handle cancellations properly
        # Wrap in gather with return_exceptions=True to ensure all are attempted
        results = await asyncio.gather(*([feeder_task] if feeder_task else []) + all_processing_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            task_name = "Feeder" if i == 0 and feeder_task else all_processing_tasks[i-(1 if feeder_task else 0)].get_name()
            if isinstance(result, asyncio.CancelledError):
                logger.info(f"Task {task_name} was cancelled successfully.")
            elif isinstance(result, Exception):
                logger.error(f"Task {task_name} raised an exception during cleanup/await: {result}", exc_info=result)


        if download_session:
            await download_session.close()
            logger.info("AIOHTTP download session closed.")

        for vw in video_writer_dict.values():
            if vw: vw.release()
        logger.info("Video writers released.")
        
        pbar_feed.close()
        pbar_preprocess.close()
        pbar_gpu_infer.close()
        pbar_postprocess.close()

        gpu_controller.stop_monitoring() # Stop the thread

        del model
        del image_processor_hf
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Main pipeline cleanup complete.")
        end_time_main = time.time()
        logger.info(f"Total execution time: {end_time_main - start_time_main:.2f} seconds.")

        # --- Audio Remuxing for Video ---
        if args.input_type == 'video' and original_video_path_for_audio and os.path.exists(depth_video_only_path):
            logger.info(f"Attempting to merge audio from {original_video_path_for_audio} into {depth_video_only_path}, output to {final_output_video_path}")
            # Ensure final_output_video_path is different or depth_video_only_path will be overwritten before reading
            # It's better to output to a new file.
            ffmpeg_cmd = [
                'ffmpeg', '-y', # -y overwrites output file without asking
                '-i', depth_video_only_path,    # Input depth video (no audio)
                '-i', original_video_path_for_audio, # Input original video (with audio)
                '-c:v', 'copy',          # Copy video stream as is
                '-c:a', 'aac',           # Re-encode audio to AAC (common, compatible)
                '-shortest',             # Finish encoding when the shortest input stream ends
                '-map', '0:v:0',         # Map video from first input
                '-map', '1:a:0',         # Map audio from second input (select first audio stream)
                final_output_video_path
            ]
            try:
                logger.info(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
                process = await asyncio.create_subprocess_exec(
                    *ffmpeg_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                if process.returncode == 0:
                    logger.info(f"FFmpeg successfully merged audio. Output: {final_output_video_path}")
                    # Optionally remove the depth_video_only_path
                    # os.remove(depth_video_only_path)
                else:
                    logger.error(f"FFmpeg failed with code {process.returncode}.")
                    if stdout: logger.error(f"FFmpeg stdout: {stdout.decode(errors='ignore')}")
                    if stderr: logger.error(f"FFmpeg stderr: {stderr.decode(errors='ignore')}")
            except FileNotFoundError:
                logger.error("FFmpeg command not found. Please install FFmpeg and ensure it's in your PATH.")
            except Exception as e:
                logger.error(f"Error during FFmpeg execution: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='(Async) Generate depth maps with GPU/CPU usage control.')
    
    parser.add_argument('--input_type', type=str, choices=['folder', 'url_file', 'video'], required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--resolution_scale', type=float, default=1.0)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--batch_size_min', type=int, default=1)
    parser.add_argument('--batch_size_max', type=int, default=4, help="Max batch size GPU inferencer will *try* to make.")

    parser.add_argument('--target_gpu_util', type=int, default=60)
    parser.add_argument('--gpu_max_allowed_util', type=int, default=78)
    parser.add_argument('--gpu_power_limit', type=int, default=None)
    parser.add_argument('--memory_fraction', type=float, default=0.75)

    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--quantized', action='store_true')
    
    args = parser.parse_args()

    if args.batch_size_min < 1: exit("Error: --batch_size_min must be >= 1.")
    if args.batch_size_max < args.batch_size_min: exit(f"Error: --batch_size_max must be >= --batch_size_min.")
    if args.gpu_max_allowed_util <= args.target_gpu_util: logger.warning(f"--gpu_max_allowed_util should be > --target_gpu_util.")

    try:
        asyncio.run(run_pipeline(args))
    except KeyboardInterrupt:
        logger.info("Main program interrupted by user. Exiting.")
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}", exc_info=True)

