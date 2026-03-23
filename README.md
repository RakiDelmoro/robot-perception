# Robot Perception - Real-time Video Captioning

Real-time video captioning system using Qwen2.5-VL GGUF model with GPU acceleration.

## Architecture

```
[Webcam] → [Flask Server (Host)] → [MJPEG Stream] → [Container: Caption Model] → [Web UI]
```

- **Host**: Flask MJPEG server streaming webcam
- **Container**: Qwen2.5-VL GGUF model for vision-language inference
- **Web UI**: Live video with smooth caption overlay

## Requirements

- **GPU**: NVIDIA with CUDA support
- **Python**: 3.10+
- **Docker** with GPU passthrough

## Setup

### 1. Install llama-cpp-python with CUDA

```bash
pip install llama-cpp-python --force-reinstall --no-cache-dir \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

### 2. Prepare Model Files

Place in `/workspaces/robot-perception/`:
- `Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf` (main model)
- `mmproj-BF16.gguf` (vision projector)

Download from: https://huggingface.co/unsloth/Qwen2.5-VL-3B-Instruct-GGUF

### 3. Run on Host (Webcam Server)

```bash
python webcam_server.py
```

### 4. Run in Container

```bash
python realtime_caption.py
```

### 5. Open in Browser

```
http://localhost:5000
```

## Features

| Feature | Description |
|---------|-------------|
| GPU Acceleration | CUDA-enabled llama-cpp-python |
| Async Pipeline | Non-blocking frame capture |
| Smooth Transitions | Fade in/out caption updates |
| Low Latency | ~3s caption generation |

## Command Options

```bash
python realtime_caption.py \
  --stream-url http://host.docker.internal:8080/video_feed \
  --frame-skip 8 \
  --max-tokens 64 \
  --temperature 0.0 \
  --port 5000
```

| Option | Default | Description |
|--------|---------|-------------|
| `--stream-url` | host.docker.internal:8080 | MJPEG stream source |
| `--frame-skip` | 8 | Process every N frames |
| `--max-tokens` | 64 | Max caption length |
| `--temperature` | 0.0 | Generation temperature |
| `--port` | 5000 | Web server port |

## Performance

| Metric | Value |
|--------|-------|
| GPU Utilization | ~98% |
| Vision Encode | ~2.8s |
| Text Generation | ~5-10s |
| Frame Capture | 100+ FPS |

## Files

```
.
├── webcam_server.py       # Flask MJPEG server (run on host)
├── webcam_client.py       # MJPEG client for container
├── model_inference.py    # Qwen2.5-VL GGUF loader
├── realtime_caption.py    # Main server + pipeline
├── templates/index.html   # Web UI
├── Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf
└── mmproj-BF16.gguf
```
