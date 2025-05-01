# Depth Map Generator

A high-performance, resource-efficient depth map generation tool that can process images and videos while maintaining controlled GPU/CPU usage. Built with PyTorch and optimized for background operation.

## Features

- 🚀 Fast depth map generation for images and videos
- 🎯 Precise GPU/CPU usage control (stays below 50% by default)
- 🔄 Background operation with low system impact
- 🎞️ Support for both images and videos
- 🖼️ Batch processing with dynamic queue system
- 💾 Memory-efficient operation
- 🛠️ Automatic resource management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/depth_map_gen.git
cd depth_map_gen
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python gen.py --input_type [folder|url_file|video] --input_path PATH --output_dir OUTPUT_DIR
```

### Advanced Options

```bash
python gen.py \
    --input_type [folder|url_file|video] \
    --input_path PATH \
    --output_dir OUTPUT_DIR \
    --max_frames MAX_FRAMES \
    --resolution_scale SCALE \
    --target_gpu_util UTILIZATION \
    --gpu_power_limit WATTS \
    --quantized
```

### Parameters

- `--input_type`: Type of input (folder, url_file, or video)
- `--input_path`: Path to input file/folder
- `--output_dir`: Directory to save output
- `--max_frames`: Maximum frames to process (for videos)
- `--resolution_scale`: Scale factor for input resolution (default: 1.0)
- `--target_gpu_util`: Target GPU utilization percentage (default: 50)
- `--gpu_power_limit`: GPU power limit in watts
- `--quantized`: Use 4-bit quantized model for reduced memory usage

## How It Works

### Resource Management

The application uses a sophisticated resource management system:

1. **GPU Controller**: Monitors and controls GPU usage using a PID controller
2. **Memory Management**: Automatic memory cleanup and optimization
3. **Process Priority**: Runs at low priority to minimize system impact
4. **Dynamic Batching**: Adjusts batch size based on resource usage

### Processing Pipeline

1. **Input Handling**: Supports images, video frames, and URLs
2. **Preprocessing**: Fast image processing with optimized settings
3. **Inference**: Efficient depth estimation using the Depth-Anything model
4. **Postprocessing**: Depth map normalization and saving
5. **Resource Monitoring**: Continuous monitoring of GPU/CPU usage

### Queue System

The application implements a dynamic queue system that:
- Processes multiple images simultaneously (up to 5)
- Automatically adjusts batch size based on resource usage
- Maintains controlled GPU/CPU utilization
- Optimizes memory usage

## Performance Optimization

- Uses fast image processor
- Implements efficient memory management
- Supports 4-bit quantization
- Optimized tensor operations
- Dynamic batch processing

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch with CUDA support
- See requirements.txt for full dependencies

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 