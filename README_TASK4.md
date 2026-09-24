# Task 4 - Object Detection and Tracking

This project satisfies the Task 4 requirements using **YOLOv8 + Deep SORT + OpenCV**.

## Requirements covered

1. **Real-time video input** - OpenCV reads `walkers.mp4`. Change `VIDEO_PATH` to `0` in `tracking_ppl.py` to use a webcam.
2. **Pre-trained object detection** - YOLOv8n (`yolov8n.pt`) detects objects in every frame.
3. **Bounding boxes and labels** - Detected objects are drawn with class labels and confidence filtering.
4. **Object tracking** - Deep SORT maintains persistent IDs across frames.
5. **Real-time output** - OpenCV displays bounding boxes, labels, tracking IDs and movement trails.

## Installation

Activate your virtual environment and run:

```powershell
pip install -r requirements.txt
```

## Run Task 4

```powershell
python tracking_ppl.py
```

Press **Q** to stop the video.

## Webcam option

In `tracking_ppl.py`, change:

```python
VIDEO_PATH = "walkers.mp4"
```

to:

```python
VIDEO_PATH = 0
```

Then run the same command.

## Pipeline

**OpenCV video → YOLOv8 detection → Deep SORT tracking → bounding boxes + class labels + persistent IDs → OpenCV display**
