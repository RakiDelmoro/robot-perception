# Temporal Streaming Pipeline

Event-driven robot perception with temporal understanding.

## Quick Start

```bash
python streaming_wrapper/integration/stream_pipeline.py --stream-url http://host.docker.internal:8080/video_feed
python streaming_wrapper/integration/stream_pipeline.py --source webcam
python streaming_wrapper/integration/stream_pipeline.py --no-vlm
```

Open http://localhost:5000/

## Architecture

```
[Video] → [Detect] → [Track] → [State] → [Predict] → [Events] → [VLM] → [Web]
   YOLOv8      SORT     Kalman   Trajectory   Change      Qwen2.5-VL  Flask
```

**Key insight**: VLM itself is NOT temporal. Temporal understanding comes from the pipeline.

## Action-Stabilized Captioning

V-JEPA approach: red=uncertainty → green=stabilized

1. Object starts moving → `in_progress` (red)
2. Object stabilizes for 3+ frames → `stable` (green)
3. VLM called to describe "what happened"

| Event | When |
|-------|------|
| `new_object` | Object appears |
| `object_left` | Object leaves |
| `action_complete` | Action stabilizes |

## Components

| File | Purpose |
|------|---------|
| `streaming_vlm.py` | Main pipeline |
| `detector.py` | YOLOv8 detection |
| `tracker.py` | SORT + Kalman |
| `state_estimator.py` | Velocity/motion |
| `predictor.py` | Trajectory prediction |
| `event_generator.py` | Action phase events |
| `websocket_server.py` | Web UI |
| `prompts.py` | VLM prompts |
| `integration/stream_pipeline.py` | Entry point |

## Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--stream-url` | host.docker.internal:8080 | MJPEG source |
| `--source` | stream | stream/webcam/video |
| `--device` | cuda | YOLO device |
| `--frame-skip` | 1 | Process every N frames |
| `--max-tracked` | 10 | Max objects |
| `--prediction-horizon` | 2.0 | Prediction window (s) |
| `--no-vlm` | - | Disable VLM |
| `--max-tokens` | 32 | Caption length |
| `--port` | 5000 | Web port |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Web UI |
| `/video_feed` | MJPEG stream |
| `/api/state` | Scene state JSON |
| `/api/events` | Event history |
| `/api/caption` | Latest caption |
| `/api/stats` | Latency stats |
