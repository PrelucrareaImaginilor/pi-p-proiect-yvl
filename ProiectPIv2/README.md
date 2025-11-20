# Person Re-Identification System

A real-time multi-person tracking and re-identification system using YOLOv8 object detection. The system detects people in video streams, tracks them across frames, and maintains persistent identities even when subjects temporarily leave the frame.

## Features

- Real-time person detection using YOLOv8
- Persistent ID tracking across frame exits/re-entries
- HSV color histogram-based feature extraction
- Configurable similarity thresholds and memory management
- Support for video files and webcam input
- Headless mode for server deployment
- Video output saving

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Process a video file
python main.py video.mp4

# Use webcam (camera index 0)
python main.py 0
```

### Command-Line Options

```bash
python main.py [video] [options]

Arguments:
  video                 Path to video file or camera index (default: peopleWalking.mp4)

Options:
  --model MODEL         Path to YOLO model (default: yolov8s.pt)
  --config CONFIG       Path to configuration file (default: config.yaml)
  --output, -o OUTPUT   Save output video to file
  --no-display          Run without display window (headless mode)
  --scale SCALE         Frame scale factor (default: 0.4)
  --verbose, -v         Enable verbose logging
```

### Examples

```bash
# Process video with output
python main.py input.mp4 -o output.mp4

# Run headless with custom model
python main.py video.mp4 --model yolov8m.pt --no-display -o result.mp4

# Use webcam with verbose logging
python main.py 0 -v

# Custom scale factor
python main.py video.mp4 --scale 0.6
```

## Configuration

Edit `config.yaml` to customize:

```yaml
app_config:
  model_name: yolov8s.pt
  target_class: 0  # person class
  confidence_thresh: 0.5
  reid_similarity_thresh: 0.75
  max_frames_missing: 3000
  feature_history_size: 10
  cleanup_interval: 1000
  frame_scale: 0.4
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `reid_similarity_thresh` | Minimum similarity for ID matching | 0.75 |
| `max_frames_missing` | Frames before ID cleanup | 3000 |
| `feature_history_size` | Features stored per person | 10 |
| `cleanup_interval` | Frames between cleanup cycles | 1000 |
| `confidence_thresh` | YOLO detection confidence | 0.5 |

## How It Works

1. **Detection**: YOLOv8 detects people in each frame
2. **Feature Extraction**: HSV color histograms are computed for each detection
3. **Matching**: New detections are matched against known IDs using:
   - Feature similarity (50%)
   - Maximum historical similarity (20%)
   - Spatial proximity (IoU) (20%)
   - Temporal recency (10%)
4. **ID Management**: Unmatched detections get new IDs; old IDs are cleaned up

## Project Structure

```
ProiectPIv2/
├── main.py           # Main application
├── config.yaml       # Application configuration
├── memory.yaml       # ByteTrack tracker configuration
├── requirements.txt  # Dependencies
├── yolov8s.pt        # YOLO model (download separately)
└── README.md         # This file
```

## Controls

- Press `q` to exit
- Press `Ctrl+C` to interrupt

## Limitations

- HSV histograms are sensitive to lighting changes
- Similar clothing colors may cause ID switches
- Performance depends on GPU availability

## License

MIT License
