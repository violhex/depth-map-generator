import os
import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse
import requests
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers import BitsAndBytesConfig # Explicit import needed for quantized
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

# GPU Monitoring Support
GPU_MONITOR_AVAILABLE = False
try:
    from pynvml import (
        nvmlInit, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates, nvmlShutdown, NVMLError
    )
    nvmlInit()
    GPU_MONITOR_AVAILABLE = True
    logging.info("pynvml initialized successfully. GPU monitoring enabled.")
except ImportError:
    logging.warning("pynvml not found. GPU monitoring disabled.")
except NVMLError as e:
    logging.warning(f"pynvml initialization failed: {e}. GPU monitoring disabled.")
except Exception as e:
    logging.error(f"Unexpected error during pynvml initialization: {e}")


# CPU Throttling & Resource Limits
if os.name == 'posix':
    import resource

# -- GPU Controller Class --
class GPUController:
    def __init__(self, user_target_utilization=70, max_allowed_gpu_util=78, polling_interval=0.5,
                 pid_kp=0.05, pid_ki=0.005, pid_kd=0.02, max_batch_size=4, min_batch_size=1):
        self.user_target_utilization = user_target_utilization
        self.max_allowed_gpu_util = max_allowed_gpu_util
        # PID will try to keep utilization lower to avoid hitting max_allowed_gpu_util
        self.pid_target_util = min(user_target_utilization, self.max_allowed_gpu_util - 5)
        self.pid_target_util = max(30, self.pid_target_util) # Ensure PID target is not too low

        self.polling_interval = polling_interval
        self.current_gpu_util = 0
        self.stop_event = threading.Event()
        self.active = False # Becomes true if pynvml initializes correctly
        self.adaptive_sleep_time = 0.05 # Start with a small sleep
        self.max_sleep_time = 2.0 # Max sleep PID can impose
        
        # Ensure batch sizes are valid
        self.min_batch_size = max(1, min_batch_size)
        self.max_batch_size = max(self.min_batch_size, max_batch_size)
        self.batch_size = self.min_batch_size # Start with min batch size
        
        self.handle = None
        if GPU_MONITOR_AVAILABLE:
            try:
                self.handle = nvmlDeviceGetHandleByIndex(0) # Assuming GPU 0
                self.active = True
                self.monitor_thread = threading.Thread(target=self._monitor_resources, name="ResourceMonitorThread", daemon=True)
                self.monitor_thread.start()
                logging.info(f"GPUController initialized. PID target: {self.pid_target_util}%, Max allowed GPU: {self.max_allowed_gpu_util}%. Batch Size: {self.batch_size} (Min: {self.min_batch_size}, Max: {self.max_batch_size})")
            except NVMLError as e:
                logging.warning(f"Failed to get GPU handle: {e}. GPUController will be inactive.")
                self.active = False
            except Exception as e:
                logging.error(f"Unexpected error getting GPU handle: {e}. GPUController inactive.")
                self.active = False
        else:
            logging.info("GPUController is inactive as pynvml is not available or failed to init.")

        self.pid_control = {
            'last_error': 0,
            'integral': 0,
            'kp': pid_kp,
            'ki': pid_ki,
            'kd': pid_kd
        }
        
        self.current_cpu_util = 0
        # Use a separate CPU target if desired, or link to GPU target
        self.cpu_target = self.pid_target_util # Align CPU target for PID control for now

        self.last_memory_clear_time = time.time()
        self.memory_clear_interval = 60  # Clear memory less frequently by default

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
            if self.monitor_thread.is_alive():
                 logging.warning("Resource monitor thread did not terminate gracefully.")
        if GPU_MONITOR_AVAILABLE and self.active: # Check self.active too
            try:
                nvmlShutdown()
                logging.info("pynvml shutdown.")
            except NVMLError as e:
                logging.warning(f"Error during nvmlShutdown: {e}")
            except Exception as e:
                logging.error(f"Unexpected error during nvmlShutdown: {e}")
            
    def _monitor_resources(self):
        import psutil # For CPU monitoring
        pid = os.getpid()
        py_process = psutil.Process(pid)

        while not self.stop_event.is_set():
            try:
                # Get GPU utilization
                if self.handle:
                    util_rates = nvmlDeviceGetUtilizationRates(self.handle)
                    self.current_gpu_util = util_rates.gpu
                else:
                     self.current_gpu_util = 0 # No handle, assume 0

                # Get CPU utilization for the current process
                # self.current_cpu_util = psutil.cpu_percent(interval=None) # System-wide CPU %
                self.current_cpu_util = py_process.cpu_percent(interval=None) / psutil.cpu_count() # Per-process CPU % normalized by core count

                # Memory management
                current_time = time.time()
                if current_time - self.last_memory_clear_time > self.memory_clear_interval:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect() # Python GC
                    self.last_memory_clear_time = current_time
                    logging.debug("Periodic memory clear performed.")
                
                # PID control for throttling
                gpu_error = self.current_gpu_util - self.pid_target_util
                # Use a higher target for CPU as per-process utilization is often lower
                cpu_error = self.current_cpu_util - (self.cpu_target * 1.5) # Allow higher process CPU%
                
                # Combine errors: prioritize GPU if high, otherwise consider CPU.
                if self.current_gpu_util > self.pid_target_util + 5: # GPU clearly over target
                    effective_error = gpu_error
                else: # GPU is ok or low, consider CPU if it's high
                    effective_error = max(gpu_error, cpu_error if cpu_error > 0 else 0) # Only throttle if CPU is over target
                
                # PID terms
                p_term = self.pid_control['kp'] * effective_error
                # Integral term: accumulate positive error, decay for negative error, prevent windup
                dt = self.polling_interval # Time delta
                self.pid_control['integral'] = self.pid_control['integral'] + effective_error * dt
                integral_max = 20 / (self.pid_control['ki'] + 1e-6) # Limit integral to prevent excessive overshoot
                self.pid_control['integral'] = max(-integral_max, min(self.pid_control['integral'], integral_max)) 

                # Prevent large integral buildup when error is zero or negative
                if effective_error <= 0:
                    self.pid_control['integral'] *= 0.95 # Decay integral slowly when below target


                d_term = self.pid_control['kd'] * (effective_error - self.pid_control['last_error']) / dt
                self.pid_control['last_error'] = effective_error
                
                # Apply PID output to sleep time
                pid_output = p_term + self.pid_control['ki'] * self.pid_control['integral'] + d_term
                if pid_output > 0: # Over target or needs slowing down
                    self.adaptive_sleep_time = min(self.max_sleep_time, max(0.01, pid_output))
                else: # Under target, reduce sleep time
                    decay_factor = 0.95 if effective_error > -10 else 0.85 # Faster decay if well below target
                    self.adaptive_sleep_time = max(0.0, self.adaptive_sleep_time * decay_factor)
                
                # Batch size control (adjust based on PID and direct utilization)
                combined_util = max(self.current_gpu_util, self.current_cpu_util)
                
                if combined_util > self.max_allowed_gpu_util: # Hard cap breached
                    self.batch_size = self.min_batch_size # Immediate drop
                elif combined_util > self.pid_target_util + 10: # Significantly over target
                     if self.batch_size > self.min_batch_size:
                         self.batch_size = max(self.min_batch_size, int(self.batch_size * 0.9)) # Gentle decrease
                elif combined_util < self.pid_target_util - 15: # Well below target
                    if self.adaptive_sleep_time < 0.02 : # Only increase if not sleeping much
                         if self.batch_size < self.max_batch_size:
                             self.batch_size = min(self.max_batch_size, self.batch_size + 1)
                
                # Ensure batch size is an integer
                self.batch_size = int(round(self.batch_size))


                logging.debug(
                    f"GPU: {self.current_gpu_util:.1f}% (PID Target: {self.pid_target_util:.1f}%, Max: {self.max_allowed_gpu_util:.1f}%) | "
                    f"CPU: {self.current_cpu_util:.1f}% (PID Target: {self.cpu_target:.1f}%) | "
                    f"Sleep: {self.adaptive_sleep_time:.3f}s | Batch: {self.batch_size} | "
                    f"Err: {effective_error:.1f}, P: {p_term:.2f}, I: {self.pid_control['ki'] * self.pid_control['integral']:.2f}, D: {d_term:.2f}"
                )
                
            except NVMLError as e:
                logging.warning(f"pynvml error in monitor thread: {e}. GPU monitoring might be compromised.")
                # self.active = False # Optionally disable if NVML errors persist
            except psutil.NoSuchProcess:
                logging.warning("Monitor thread: Process not found. Exiting monitor.")
                break
            except Exception as e:
                logging.error(f"Unexpected error in _monitor_resources: {e}", exc_info=True)
            
            # Wait for the next polling interval, but wake up early if stop event is set
            self.stop_event.wait(timeout=self.polling_interval)

    def wait_for_gpu(self):
        if not self.active:
            time.sleep(0.01) # Small sleep even if inactive to prevent busy-waiting loops elsewhere
            return
        
        initial_wait_logged = False
        # Check against MAX allowed threshold
        while (self.current_gpu_util > self.max_allowed_gpu_util) or \
              (self.current_cpu_util > self.max_allowed_gpu_util): # Use max_allowed for CPU too as a hard limit

            if not initial_wait_logged:
                logging.info( # Log once when waiting starts
                    f"Load exceeds max allowed ({self.max_allowed_gpu_util}%). Waiting... "
                    f"GPU: {self.current_gpu_util:.1f}%, CPU: {self.current_cpu_util:.1f}%."
                )
                initial_wait_logged = True

            # Use the adaptive sleep time calculated by PID, plus a base minimum wait
            sleep_duration = max(self.adaptive_sleep_time, 0.1) # Ensure at least 0.1s sleep when waiting
            sleep_duration_jittered = sleep_duration * (0.9 + 0.2 * np.random.random()) # Add jitter
            
            logging.debug(f"Waiting... sleeping for {sleep_duration_jittered:.3f}s (Adaptive: {self.adaptive_sleep_time:.3f}s)")
            
            # Wait, but check stop event periodically
            interrupted = self.stop_event.wait(timeout=sleep_duration_jittered)
            if interrupted:
                logging.info("Stop event received during wait_for_gpu.")
                break # Exit loop if stop event is set

            # Optional: Clear memory more frequently when waiting? Maybe not necessary.
            # current_time = time.time()
            # if current_time - self.last_memory_clear_time > self.memory_clear_interval / 2:
            #     if torch.cuda.is_available(): torch.cuda.empty_cache()
            #     gc.collect()
            #     self.last_memory_clear_time = current_time
        
        if initial_wait_logged: # Log when waiting finishes
            logging.info(f"Load acceptable. Resuming. GPU: {self.current_gpu_util:.1f}%, CPU: {self.current_cpu_util:.1f}%")


    def is_processing_critically_loaded(self, threshold_offset=5):
        """Checks if current load is close to the maximum allowed limit."""
        if not self.active:
            return False
        gpu_critical = self.current_gpu_util > (self.max_allowed_gpu_util - threshold_offset)
        cpu_critical = self.current_cpu_util > (self.max_allowed_gpu_util - threshold_offset) # Check CPU against same high threshold
        return gpu_critical or cpu_critical

# -- Utility Functions --

def set_system_resource_limits(nice_value=10):
    """Sets process priority (nice) and CPU thread limits."""
    try:
        # Set CPU thread limits for PyTorch and underlying libraries (OpenMP, MKL)
        num_cores = os.cpu_count()
        # Use a fraction of cores, e.g., half, but at least 1 and capped at maybe 4 or 8 for background tasks
        constrained_threads = max(1, min(num_cores // 2 if num_cores else 1, 4)) 
        
        torch.set_num_threads(constrained_threads)
        os.environ['OMP_NUM_THREADS'] = str(constrained_threads)
        os.environ['MKL_NUM_THREADS'] = str(constrained_threads)
        logging.info(f"Global thread limit set to {constrained_threads} for PyTorch/OMP/MKL.")

        # Set process priority (niceness)
        if os.name == 'posix':
            os.nice(nice_value)
            logging.info(f"Process priority set to nice={nice_value}")
        elif os.name == 'nt':  # Windows
            import win32api, win32process, win32con
            pid = win32api.GetCurrentProcessId()
            handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
            win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
            logging.info("Process priority set to Below Normal (Windows).")

    except ImportError:
        logging.warning("win32api not found for setting process priority on Windows.")
    except Exception as e:
        logging.warning(f"Failed to set some resource limits: {e}")

def download_image(url):
    """Downloads an image from a URL."""
    try:
        logging.debug(f"Downloading image from {url}")
        # Add headers to mimic a browser request, can sometimes help
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, timeout=20, headers=headers, stream=True) # Use stream=True for potentially large images
        response.raise_for_status()
        
        # Check content type if possible
        content_type = response.headers.get('content-type')
        if content_type and not content_type.lower().startswith('image/'):
             logging.warning(f"URL {url} returned non-image content type: {content_type}")
             # return None # Optional: be stricter

        img = Image.open(BytesIO(response.content)).convert("RGB")
        logging.info(f"Successfully downloaded image from {url} ({img.width}x{img.height})")
        return img
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error downloading {url}: {e}")
    except UnidentifiedImageError:
         logging.error(f"Cannot identify image file from {url}. Content might not be a valid image.")
    except Exception as e:
        logging.error(f"Error processing downloaded image from {url}: {e}", exc_info=True)
    return None

def save_depth_map(depth_map_image, output_path):
    """Saves a PIL Image."""
    try:
        depth_map_image.save(output_path)
        logging.info(f"Saved depth map: {output_path}")
    except Exception as e:
        logging.error(f"Error saving depth map to {output_path}: {e}")

def save_depth_frame(depth_array_uint8, output_video_writer):
    """Converts grayscale uint8 numpy array to BGR and writes to video writer."""
    try:
        # Ensure it's single channel grayscale before converting
        if len(depth_array_uint8.shape) == 3 and depth_array_uint8.shape[2] == 3:
             # Already BGR? Assume it's correct.
             depth_bgr = depth_array_uint8
        elif len(depth_array_uint8.shape) == 2:
             # Grayscale, convert
             depth_bgr = cv2.cvtColor(depth_array_uint8, cv2.COLOR_GRAY2BGR)
        else:
             logging.error(f"Invalid shape for depth frame: {depth_array_uint8.shape}")
             return
        output_video_writer.write(depth_bgr)
    except Exception as e:
        logging.error(f"Error writing depth frame: {e}")

# -- Image Processing --

class ProcessingQueue:
    """Simple thread-safe bounded queue."""
    def __init__(self, max_size=10):
        self.queue = queue.Queue(maxsize=max_size)
        self.max_size = max_size
        
    def put(self, item, block=True, timeout=None):
        try:
            self.queue.put(item, block=block, timeout=timeout)
            return True
        except queue.Full:
            return False
            
    def get(self, block=True, timeout=None):
        try:
            return self.queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
            
    def size(self):
        return self.queue.qsize()
            
    def is_full(self):
        return self.queue.full()
            
    def is_empty(self):
        return self.queue.empty()

def process_and_save_image(image, filename, output_dir, image_processor, model, device, gpu_controller, resolution_scale=1.0):
    """Processes a single PIL image and saves the depth map."""
    inputs_gpu = None
    predicted_depth_gpu = None
    prediction_gpu = None
    original_pil_image_size = image.size # Store original size before potential scaling
    
    try:
        # 1. Preprocessing (Resizing)
        target_pil_image = image
        if resolution_scale != 1.0:
            try:
                w, h = image.size
                new_size = (int(w * resolution_scale), int(h * resolution_scale))
                target_pil_image = image.resize(new_size, Image.Resampling.LANCZOS)
                logging.debug(f"Resized image {filename} from {image.size} to {new_size} for processing.")
            except Exception as e:
                logging.error(f"Error resizing image {filename}: {e}")
                del image # Clean up original
                return # Cannot proceed if resizing fails
        
        # Store the size of the image actually fed to the processor
        processed_image_size = target_pil_image.size

        # 2. Prepare Model Input
        with torch.no_grad(): # Image processor might use torch ops
            inputs_cpu = image_processor(images=target_pil_image, return_tensors='pt')
        
        # Move input to GPU
        inputs_gpu = {k: v.to(device) for k, v in inputs_cpu.items()}
        del inputs_cpu, target_pil_image, image # Free CPU memory early

        # 3. Inference
        if gpu_controller: gpu_controller.wait_for_gpu() # Wait before inference
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(**inputs_gpu)
            predicted_depth_gpu = outputs.predicted_depth # Shape: [1, H_model, W_model]
        
        # 4. Clean up GPU inputs and intermediate outputs
        del inputs_gpu
        if hasattr(outputs, 'hidden_states'): del outputs.hidden_states
        if hasattr(outputs, 'attentions'): del outputs.attentions
        del outputs
        if device.type == 'cuda': torch.cuda.empty_cache() # Clean cache after model use

        # 5. Post-processing (Interpolation)
        # Interpolate to the original input image size (before scaling)
        target_interpolation_size = (original_pil_image_size[1], original_pil_image_size[0]) # H, W format

        with torch.no_grad():
            prediction_gpu = F.interpolate(
                predicted_depth_gpu.unsqueeze(1), # Add channel dim: [1, 1, H_model, W_model]
                size=target_interpolation_size,   # Target H, W 
                mode="bicubic",
                align_corners=False
            )
        del predicted_depth_gpu # Free model output tensor
        
        # 6. Move result to CPU and Normalize
        depth_cpu_numpy = prediction_gpu.squeeze().to(torch.float32).cpu().numpy()
        del prediction_gpu # Free interpolated tensor
        if device.type == 'cuda': torch.cuda.empty_cache() # Final GPU cleanup for this image

        depth_min = depth_cpu_numpy.min()
        depth_max = depth_cpu_numpy.max()
        if depth_max - depth_min > 1e-6:
            depth_normalized = (depth_cpu_numpy - depth_min) / (depth_max - depth_min) * 255.0
        else:
            depth_normalized = np.zeros_like(depth_cpu_numpy)
            logging.warning(f"Depth map for {filename} had near-zero range, saving as black.")
        
        depth_uint8 = depth_normalized.astype(np.uint8)
        depth_image_pil = Image.fromarray(depth_uint8)

        # 7. Save Output
        base_name = os.path.splitext(filename)[0] if isinstance(filename, str) else f"image_{time.time_ns()}"
        output_filename = f"depth_{base_name}.png"
        output_path = os.path.join(output_dir, output_filename)
        save_depth_map(depth_image_pil, output_path)
        
        # 8. Final CPU Cleanup
        del depth_cpu_numpy, depth_normalized, depth_uint8, depth_image_pil
        gc.collect()

    except Exception as e:
        logging.error(f"Error processing image '{filename}': {e}", exc_info=True)
        # Attempt to clean up any potentially lingering GPU tensors
        del inputs_gpu, predicted_depth_gpu, prediction_gpu
        if device.type == 'cuda': torch.cuda.empty_cache()
        gc.collect()
        return # Explicitly return on error


def process_folder(input_path, output_dir, image_processor, model, device, gpu_controller, resolution_scale=1.0):
    """Processes all images in a folder using worker threads."""
    all_files = os.listdir(input_path)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'))]
    if not image_files:
        logging.info(f"No compatible images found in {input_path}")
        return

    # Dynamic queue size and worker count based on GPU controller state? Could be complex. Stick to simpler logic for now.
    queue_max_size = max(5, (gpu_controller.max_batch_size * 2) if (gpu_controller and gpu_controller.active) else 5)
    img_processing_queue = ProcessingQueue(max_size=queue_max_size) 
    
    num_workers = max(1, min(os.cpu_count() // 2 if os.cpu_count() else 1, 4)) # Conservative worker count
    logging.info(f"Starting {num_workers} worker threads for image processing. Queue size: {queue_max_size}")

    stop_worker_event = threading.Event() # Separate event for worker termination

    def worker_task():
        while not stop_worker_event.is_set():
            try:
                # Get from queue with a timeout to allow checking the stop event
                item = img_processing_queue.get(block=True, timeout=0.5) 
                if item is None: # Sentinel value means no more items will be added
                    img_processing_queue.put(None) # Put sentinel back for other workers
                    break # Exit loop normally
                
                image, filename = item
                logging.debug(f"Worker {threading.current_thread().name} starting {filename}")
                process_and_save_image(image, filename, output_dir, image_processor, model, device, gpu_controller, resolution_scale)
                del image # Ensure image object is released after processing
                logging.debug(f"Worker {threading.current_thread().name} finished {filename}")
                img_processing_queue.task_done() # Mark task as complete for queue joining
                gc.collect() # Optional: Collect garbage periodically in worker

            except queue.Empty:
                # Queue is empty, just continue loop and check stop_event again
                continue 
            except Exception as e:
                 logging.error(f"Error in worker thread {threading.current_thread().name}: {e}", exc_info=True)
                 # Ensure task_done is called even if an error occurs processing an item
                 try: img_processing_queue.task_done() 
                 except ValueError: pass # Ignore if task_done called multiple times

        logging.info(f"Worker thread {threading.current_thread().name} finishing.")


    worker_threads = [threading.Thread(target=worker_task, name=f"ImgWorker-{i}", daemon=True) for i in range(num_workers)]
    for t in worker_threads:
        t.start()
    
    # --- Queueing Loop ---
    queued_count = 0
    skipped_count = 0
    try:
        for filename in tqdm(image_files, desc="Queueing images"):
            if stop_worker_event.is_set(): # Check if stop was requested during queueing
                logging.info("Stop requested during queueing, stopping image loading.")
                break
            
            full_path = os.path.join(input_path, filename)
            try:
                image = Image.open(full_path).convert("RGB")
                if image.width <= 0 or image.height <= 0:
                    logging.warning(f"Invalid image dimensions for {filename} ({image.width}x{image.height}), skipping.")
                    skipped_count += 1
                    continue
                
                # Wait if queue is full OR if GPU/CPU is critically loaded to avoid overwhelming the system
                wait_start_time = None
                while img_processing_queue.is_full() or \
                      (gpu_controller and gpu_controller.is_processing_critically_loaded()):
                    if wait_start_time is None:
                         wait_start_time = time.time()
                         logging.debug("Queue full or system critically loaded, pausing queueing...")
                    time.sleep(0.2) # Check fairly frequently
                    if gpu_controller and gpu_controller.stop_event.is_set(): # Check main stop event
                         stop_worker_event.set() # Signal workers to stop
                         raise KeyboardInterrupt("Main stop event detected while waiting to queue.")
                    if stop_worker_event.is_set(): # Check worker stop event
                         raise KeyboardInterrupt("Worker stop event detected while waiting to queue.")
                
                if wait_start_time is not None:
                    logging.debug(f"Resuming queueing after waiting {time.time() - wait_start_time:.2f}s.")

                # Put image and filename onto the queue
                if not img_processing_queue.put((image, filename), timeout=5): # Put with timeout
                    logging.warning(f"Timeout adding {filename} to queue, skipping.")
                    skipped_count += 1
                    del image # Clean up image if skipped
                    continue
                queued_count += 1
                # image object ownership transferred to queue

            except FileNotFoundError:
                 logging.warning(f"Image file not found (possibly moved/deleted?): {full_path}, skipping.")
                 skipped_count += 1
            except UnidentifiedImageError:
                 logging.warning(f"Could not identify image format for {filename}, skipping.")
                 skipped_count += 1
            except Exception as e:
                logging.warning(f"Error opening or queueing image {filename}: {e}", exc_info=True)
                skipped_count += 1
                if 'image' in locals(): del image # Clean up local image var
                continue

        logging.info(f"Finished queueing {queued_count} images ({skipped_count} skipped). Waiting for processing...")
        
        # Signal queue completion by putting None sentinel
        img_processing_queue.put(None) 

        # Wait for all tasks in the queue to be processed
        img_processing_queue.join() # Wait until task_done() called for all items
        logging.info("All queued image tasks processed by workers.")

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Stopping workers and queueing...")
        stop_worker_event.set() # Signal workers to stop processing more items

    finally:
        # Ensure workers are properly joined
        stop_worker_event.set() # Make sure event is set
        logging.info("Waiting for worker threads to terminate...")
        for t in worker_threads:
            t.join(timeout=10) # Give workers time to finish current item
            if t.is_alive():
                logging.warning(f"Worker thread {t.name} did not terminate gracefully.")
        logging.info("Folder processing routine complete.")


# -- Video Processing --

def process_video_frames_batch(batch_data, image_processor, model, device, gpu_controller, 
                               output_video_writer, target_h, target_w, resolution_scale):
    """Processes a batch of video frames ([ (idx, frame_bgr), ... ])."""
    if not batch_data:
        return 0 # No frames processed

    processed_in_batch = 0
    pil_images = []
    frame_indices = [item[0] for item in batch_data] # Keep track of original indices if needed

    # 1. Preprocessing (Convert to PIL, Optional Resize)
    try:
        for frame_idx, frame_bgr in batch_data:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            if resolution_scale != 1.0: # Scale image before feeding to model
                scaled_w = int(pil_image.width * resolution_scale)
                scaled_h = int(pil_image.height * resolution_scale)
                if scaled_w > 0 and scaled_h > 0:
                    pil_image = pil_image.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
                else:
                    logging.warning(f"Skipping frame {frame_idx} due to invalid resize dimensions ({scaled_w}x{scaled_h})")
                    continue # Skip this frame if resize is invalid
            pil_images.append(pil_image)
    except Exception as e:
        logging.error(f"Error during video frame preprocessing: {e}", exc_info=True)
        del batch_data, pil_images # Clean up
        return 0 # Indicate no frames processed

    del batch_data # Free memory for raw frames BGR data

    if not pil_images: 
        logging.warning("No valid PIL images created from batch.")
        return 0

    inputs_gpu = None
    predicted_depths_gpu = None
    interpolated_depths_gpu = None

    try:
        # 2. Prepare Model Input Batch
        with torch.no_grad(): 
            # Use the image_processor's batch capabilities
            inputs_cpu = image_processor(images=pil_images, return_tensors='pt')
        del pil_images # Free PIL image list

        # Move input batch to GPU
        inputs_gpu = {k: v.to(device) for k, v in inputs_cpu.items()}
        del inputs_cpu

        # 3. Inference
        if gpu_controller: gpu_controller.wait_for_gpu() # Wait before inference

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(**inputs_gpu)
            predicted_depths_gpu = outputs.predicted_depth # Shape: [batch, H_model, W_model]
        
        # 4. Clean up GPU inputs and intermediate model outputs
        del inputs_gpu
        if hasattr(outputs, 'hidden_states'): del outputs.hidden_states
        if hasattr(outputs, 'attentions'): del outputs.attentions
        del outputs
        if device.type == 'cuda': torch.cuda.empty_cache()

        # 5. Post-processing (Interpolation)
        # Interpolate to the original video frame size (target_h, target_w)
        with torch.no_grad():
            interpolated_depths_gpu = F.interpolate(
                predicted_depths_gpu.unsqueeze(1),      # Add C dim: [N, 1, H_model, W_model]
                size=(target_h, target_w),              # Target original video frame H, W
                mode="bicubic",
                align_corners=False
            )
        del predicted_depths_gpu # Free model output tensor
        
        # 6. Move results to CPU, Normalize, Save Frame
        # Process results one by one to keep CPU memory lower
        for depth_tensor_gpu in interpolated_depths_gpu:
            depth_cpu_numpy = depth_tensor_gpu.squeeze().to(torch.float32).cpu().numpy()
            
            depth_min = depth_cpu_numpy.min()
            depth_max = depth_cpu_numpy.max()
            if depth_max - depth_min > 1e-6:
                depth_normalized = (depth_cpu_numpy - depth_min) / (depth_max - depth_min) * 255.0
            else:
                depth_normalized = np.zeros_like(depth_cpu_numpy) # Handle flat depth map
            
            depth_uint8 = depth_normalized.astype(np.uint8)
            save_depth_frame(depth_uint8, output_video_writer) # Write frame to video
            processed_in_batch += 1

            del depth_cpu_numpy, depth_normalized, depth_uint8, depth_tensor_gpu # Clean up per frame on CPU
        
        del interpolated_depths_gpu # Delete the whole batch tensor after loop
        if device.type == 'cuda': torch.cuda.empty_cache() # Final GPU cleanup for this batch

    except Exception as e:
        logging.error(f"Video batch processing error (indices around {frame_indices[0]}): {e}", exc_info=True)
        # Attempt GPU cleanup on error
        del inputs_gpu, predicted_depths_gpu, interpolated_depths_gpu
        if device.type == 'cuda': torch.cuda.empty_cache()
        return 0 # Indicate error / no frames processed this batch
    finally:
        # Ensure final cleanup happens regardless of success/failure
        gc.collect() # Collect Python garbage
    
    return processed_in_batch


def process_video(video_path, output_dir, image_processor, model, device, gpu_controller, max_frames_to_process=None, resolution_scale=1.0):
    """Reads a video, processes frames in batches, and saves a depth video."""
    if not os.path.isfile(video_path):
        logging.error(f"Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0 or fps <= 0 or vid_w <= 0 or vid_h <= 0:
        logging.error(f"Invalid video properties for {video_path}: Frames {total_frames}, FPS {fps:.2f}, W {vid_w}, H {vid_h}")
        cap.release()
        return

    logging.info(f"Video Input: {vid_w}x{vid_h} @ {fps:.2f} FPS, {total_frames} frames. Processing scale: {resolution_scale}")

    # Prepare output video writer
    input_basename = os.path.basename(video_path)
    input_name_noext = os.path.splitext(input_basename)[0]
    output_video_name = f"depth_{input_name_noext}.mp4" # Default to .mp4
    output_video_path = os.path.join(output_dir, output_video_name)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    try:
        # Output size matches original video dimensions
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (vid_w, vid_h)) 
        if not out_writer.isOpened():
            raise IOError(f"Failed to initialize VideoWriter for {output_video_path}. Check codec and permissions.")
    except Exception as e:
         logging.error(f"Error initializing video writer: {e}", exc_info=True)
         cap.release()
         return

    # Determine number of frames to actually process
    frames_to_process = min(total_frames, max_frames_to_process) if max_frames_to_process is not None else total_frames
    logging.info(f"Targeting up to {frames_to_process} frames for depth processing.")

    frame_buffer = []
    total_processed_count = 0
    start_time = time.time()
    
    # Use effective batch size from GPU controller, update dynamically
    effective_batch_size = max(1, gpu_controller.batch_size if gpu_controller and gpu_controller.active else 1)

    try:
        with tqdm(total=frames_to_process, desc="Processing Video Frames", unit="frame") as pbar:
            for frame_idx in range(frames_to_process):
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"Could not read frame {frame_idx + 1}/{frames_to_process}. Stopping video processing.")
                    break
                
                frame_buffer.append((frame_idx, frame)) # Store index and frame data

                # Process when buffer is full or it's the last frame
                is_last_frame = (frame_idx == frames_to_process - 1)
                if len(frame_buffer) >= effective_batch_size or (is_last_frame and frame_buffer):
                    processed_count_batch = process_video_frames_batch(frame_buffer, image_processor, model, device, gpu_controller,
                                                                       out_writer, vid_h, vid_w, resolution_scale)
                    
                    pbar.update(len(frame_buffer)) # Update progress bar by buffer size attempted
                    total_processed_count += processed_count_batch
                    frame_buffer = [] # Clear buffer after processing

                    # Update batch size for next iteration based on controller's current value
                    effective_batch_size = max(1, gpu_controller.batch_size if gpu_controller and gpu_controller.active else 1) 

            # Process any remaining frames in the buffer if loop finished before buffer was full
            if frame_buffer:
                logging.debug(f"Processing remaining {len(frame_buffer)} frames...")
                processed_count_batch = process_video_frames_batch(frame_buffer, image_processor, model, device, gpu_controller,
                                                                   out_writer, vid_h, vid_w, resolution_scale)
                pbar.update(len(frame_buffer)) # Update progress for the final batch
                total_processed_count += processed_count_batch

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received during video processing. Cleaning up...")
    except Exception as e:
        logging.error(f"An unexpected error occurred during video processing loop: {e}", exc_info=True)
    finally:
        # Release resources
        cap.release()
        out_writer.release()
        duration = time.time() - start_time
        logging.info(f"Finished video processing. Successfully processed and wrote {total_processed_count} depth frames in {duration:.2f} seconds.")
        if total_processed_count > 0:
             logging.info(f"Depth video saved to: {output_video_path}")
        else:
             logging.warning(f"No frames were successfully processed. Output file might be empty or invalid: {output_video_path}")
        # Final cleanup
        if device.type == 'cuda': torch.cuda.empty_cache()
        gc.collect()

# -- URL File Processing --

def process_url_file(url_file_path, output_dir, image_processor, model, device, gpu_controller, resolution_scale=1.0):
    """Reads URLs from a file, downloads images, and processes them."""
    try:
        with open(url_file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')] # Ignore empty lines and comments
    except FileNotFoundError:
        logging.error(f"URL file not found: {url_file_path}")
        return
    except Exception as e:
        logging.error(f"Error reading URL file {url_file_path}: {e}")
        return

    logging.info(f"Found {len(urls)} URLs in {url_file_path}.")
    
    for i, url in enumerate(tqdm(urls, desc="Processing URLs")):
        logging.debug(f"Processing URL {i+1}/{len(urls)}: {url}")
        image = download_image(url)
        if image is None:
            logging.warning(f"Skipping URL {url} due to download/load error.")
            continue
        
        # Create a somewhat safe filename from URL
        try:
            parsed_url = urlparse(url)
            # Use last part of path, fallback to hash
            basename = os.path.basename(parsed_url.path) if parsed_url.path else None
            if not basename: # If path is empty or just '/'
                 filename = f"url_{hash(url)}_{i}.png" # Use hash + index
            else:
                 name, ext = os.path.splitext(basename)
                 safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name) # Basic sanitization
                 filename = f"{safe_name}{ext}" if ext else f"{safe_name}.png" # Keep extension or default to png
        except Exception as e:
             logging.warning(f"Could not generate filename from URL {url}: {e}. Using default.")
             filename = f"url_default_{i}.png"

        process_and_save_image(image, filename, output_dir, image_processor, model, device, gpu_controller, resolution_scale)
        del image # Clean up downloaded image
        gc.collect() # Collect garbage after each image from URL list

    logging.info(f"Finished processing URLs from {url_file_path}.")

# -- PyTorch Resource Configuration --

def configure_pytorch_memory(max_split_mb=128):
    """Configures PyTorch CUDA memory allocator."""
    if torch.cuda.is_available():
        try:
            # Set allocator configuration using environment variable
            # 'max_split_size_mb': Reduces fragmentation by caching blocks of at least this size.
            # Smaller values might increase fragmentation but return memory to OS faster.
            # Larger values might reduce fragmentation but hold onto memory longer. 128 is a reasonable default.
            conf_val = f'max_split_size_mb:{max_split_mb}'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = conf_val
            logging.info(f"PYTORCH_CUDA_ALLOC_CONF set to '{conf_val}'.")

            # torch.cuda.set_per_process_memory_fraction(0.8) # Avoid setting a hard fraction limit unless necessary
        except Exception as e:
            logging.warning(f"Failed to configure PyTorch GPU memory settings: {e}")

def set_gpu_power_limit_watts(power_limit_watts):
    """Attempts to set GPU power limit using nvidia-smi (requires privileges)."""
    if not GPU_MONITOR_AVAILABLE:
        logging.warning("Cannot set GPU power limit: pynvml (needed for nvidia-smi check/control) is not available/active.")
        return
    try:
        # Use nvidia-smi command. This often requires root/admin privileges.
        command = ["nvidia-smi", "-pl", str(power_limit_watts)]
        logging.info(f"Attempting to run: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        logging.info(f"Successfully set GPU power limit to {power_limit_watts}W. Output: {result.stdout.strip()}")
    except FileNotFoundError:
        logging.warning("nvidia-smi command not found. Install NVIDIA drivers and ensure it's in PATH.")
    except subprocess.TimeoutExpired:
        logging.warning("nvidia-smi command timed out.")
    except subprocess.CalledProcessError as e:
        # Common error is permission denied
        logging.warning(f"Failed to set GPU power limit via nvidia-smi. Output: {e.stderr.strip()}. Requires sufficient privileges (e.g., run as admin/root).")
    except Exception as e:
        logging.error(f"An unexpected error occurred while setting GPU power limit: {e}", exc_info=True)


# -- Main Function --
def main(input_path, input_type, output_dir, max_frames=None, resolution_scale=1.0, 
         target_gpu_util=60, gpu_max_allowed_util=78, gpu_power_limit=None, quantized=False, model_name_arg=None,
         batch_size_min=1, batch_size_max=4):
    
    start_time_main = time.time()

    # 1. System Resource Limits (Priority, CPU Threads)
    set_system_resource_limits() 

    # 2. PyTorch Memory Configuration
    configure_pytorch_memory() # Use default max_split_mb

    # 3. Set GPU Power Limit (if requested and possible)
    if gpu_power_limit and torch.cuda.is_available():
        set_gpu_power_limit_watts(gpu_power_limit)

    # 4. Initialize Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if device.type == 'cuda':
        logging.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        # Clear cache before loading model
        torch.cuda.empty_cache()

    # 5. Model Selection & Loading
    model_name = model_name_arg if model_name_arg else "depth-anything/Depth-Anything-V2-Large-hf"
    logging.info(f"Loading model: {model_name} (Quantized: {quantized})")

    model = None
    image_processor = None
    try:
        # Load Image Processor
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Prepare model loading arguments
        model_load_kwargs = {"low_cpu_mem_usage": True} # Always use low CPU mem usage if possible
        
        if quantized:
            if device.type == 'cpu':
                logging.warning("BitsAndBytes 4-bit quantization is CUDA-only. Loading standard FP32 model on CPU.")
                # No special dtype for CPU load, remove if present
                model_load_kwargs.pop("torch_dtype", None) 
                model = AutoModelForDepthEstimation.from_pretrained(model_name, **model_load_kwargs)
            else: # CUDA + Quantized
                # Ensure BitsAndBytesConfig is imported
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16, # Computations done in FP16
                    bnb_4bit_quant_type="nf4",           # Use NF4 quantization
                    bnb_4bit_use_double_quant=True,      # Use double quantization for slightly better results
                )
                model_load_kwargs["quantization_config"] = quant_config
                # device_map="auto" lets accelerate handle placement for quantized models
                model_load_kwargs["device_map"] = "auto" 
                # We don't specify torch_dtype here as BNB handles it with device_map
                model = AutoModelForDepthEstimation.from_pretrained(model_name, **model_load_kwargs)
                logging.info("Loaded 4-bit quantized model with device_map='auto'.")
        
        else: # Not Quantized
            if device.type == 'cuda':
                # Load in FP16 on CUDA for non-quantized models to save memory and potentially speed up
                model_load_kwargs["torch_dtype"] = torch.float16 
                # Use device_map="auto" here too for potential multi-GPU or balancing
                model_load_kwargs["device_map"] = "auto" 
                model = AutoModelForDepthEstimation.from_pretrained(model_name, **model_load_kwargs)
                logging.info("Loaded FP16 model with device_map='auto' for CUDA.")
            else: # CPU + Not Quantized (Load in default FP32)
                 model_load_kwargs.pop("torch_dtype", None) # Ensure no FP16 dtype on CPU load
                 model = AutoModelForDepthEstimation.from_pretrained(model_name, **model_load_kwargs)
                 model = model.to(device) # Explicitly move to CPU device
                 logging.info("Loaded FP32 model on CPU.")

        # Set model to evaluation mode
        model.eval()
        
        # Clear cache again after loading, especially important on GPU
        if device.type == 'cuda':
             torch.cuda.empty_cache()
             gc.collect()
             logging.info("GPU cache cleared and GC collected after model load.")

    except ImportError as e:
        if 'bitsandbytes' in str(e).lower():
             logging.error("ImportError: bitsandbytes library not found. Please install it (`pip install bitsandbytes`) for 4-bit quantization.", exc_info=True)
        elif 'accelerate' in str(e).lower():
             logging.error("ImportError: accelerate library not found. Please install it (`pip install accelerate`) for device_map='auto'.", exc_info=True)
        else:
             logging.error(f"ImportError during model loading: {e}", exc_info=True)
        return # Cannot continue without model
    except Exception as e:
        logging.error(f"Error loading model or image processor '{model_name}': {e}", exc_info=True)
        return # Cannot continue

    # 6. Create Output Directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise OSError(f"No write access to output directory: {output_dir}")
    except OSError as e:
        logging.error(f"Error with output directory: {e}")
        return

    # 7. Initialize GPU Controller
    # Use provided args for target util, max util, and batch size constraints
    with GPUController(user_target_utilization=target_gpu_util, 
                       max_allowed_gpu_util=gpu_max_allowed_util,
                       min_batch_size=batch_size_min,
                       max_batch_size=batch_size_max) as gpu_controller:
        try:
            # 8. Select and Run Processing Function
            if input_type == 'folder':
                process_folder(input_path, output_dir, image_processor, model, device, gpu_controller, resolution_scale)
            elif input_type == 'url_file':
                process_url_file(input_path, output_dir, image_processor, model, device, gpu_controller, resolution_scale)
            elif input_type == 'video':
                process_video(input_path, output_dir, image_processor, model, device, gpu_controller, max_frames, resolution_scale)
            else:
                 # This case should not be reached due to argparse choices, but good to have
                 logging.error(f"Invalid input_type specified: {input_type}")

        except KeyboardInterrupt:
            logging.info("---- Processing interrupted by user (Ctrl+C) ----")
        except Exception as e:
            logging.error(f"An unexpected error occurred during the main processing: {e}", exc_info=True)
        finally:
            # 9. Final Cleanup (GPUController's __exit__ handles its cleanup)
            logging.info("Performing final cleanup...")
            del model # Explicitly delete model
            del image_processor
            if device.type == 'cuda':
                torch.cuda.empty_cache() # Final cache clear
            gc.collect() # Final garbage collection
            logging.info("Cleanup complete.")
            end_time_main = time.time()
            logging.info(f"Total execution time: {end_time_main - start_time_main:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate depth maps with GPU/CPU usage control and memory management.')
    
    # Input/Output Arguments
    parser.add_argument('--input_type', type=str, choices=['folder', 'url_file', 'video'], required=True, 
                        help='Type of input to process.')
    parser.add_argument('--input_path', type=str, required=True, 
                        help='Path to the input folder, URL list file, or video file.')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory where output depth maps/videos will be saved.')

    # Processing Control Arguments
    parser.add_argument('--resolution_scale', type=float, default=1.0, 
                        help='Factor to scale input resolution before processing (e.g., 0.5 for half resolution). Output depth map is resized back to original.')
    parser.add_argument('--max_frames', type=int, default=None, 
                        help='(Video only) Maximum number of frames to process from the video.')
    parser.add_argument('--batch_size_min', type=int, default=1, 
                        help='Minimum batch size for video processing (and influences worker queue). Must be >= 1.')
    parser.add_argument('--batch_size_max', type=int, default=4, 
                        help='Maximum batch size allowed by the GPU controller.')

    # Resource Control Arguments
    parser.add_argument('--target_gpu_util', type=int, default=60, 
                        help='Target GPU utilization percentage for the PID controller (e.g., 60).')
    parser.add_argument('--gpu_max_allowed_util', type=int, default=78,
                        help='Hard upper limit for GPU utilization. Processing will pause if exceeded (e.g., 78-80). Should be higher than target_gpu_util.')
    parser.add_argument('--gpu_power_limit', type=int, default=None, 
                        help='(Optional) Set GPU power limit in Watts using nvidia-smi (requires privileges).')

    # Model Arguments
    parser.add_argument('--model_name', type=str, default=None, 
                        help='(Optional) Specify a different Hugging Face model name for depth estimation.')
    parser.add_argument('--quantized', action='store_true', 
                        help='Use 4-bit quantized model (requires bitsandbytes, accelerate, and CUDA).')

    args = parser.parse_args()

    # Validate arguments
    if args.batch_size_min < 1:
         print("Error: --batch_size_min must be 1 or greater.")
         exit(1)
    if args.batch_size_max < args.batch_size_min:
         print(f"Error: --batch_size_max ({args.batch_size_max}) must be >= --batch_size_min ({args.batch_size_min}).")
         exit(1)
    if args.gpu_max_allowed_util <= args.target_gpu_util:
         print(f"Warning: --gpu_max_allowed_util ({args.gpu_max_allowed_util}%) should ideally be greater than --target_gpu_util ({args.target_gpu_util}%).")
         # Allow execution but warn user.

    # Run the main function
    main(
        input_path=args.input_path,
        input_type=args.input_type,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        resolution_scale=args.resolution_scale,
        target_gpu_util=args.target_gpu_util,
        gpu_max_allowed_util=args.gpu_max_allowed_util,
        gpu_power_limit=args.gpu_power_limit,
        quantized=args.quantized,
        model_name_arg=args.model_name,
        batch_size_min=args.batch_size_min,
        batch_size_max=args.batch_size_max
    )