import cv2
from collections import defaultdict, deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ------------------------------------------------------------
# Task 4: Object Detection and Tracking
# YOLOv8 = object detector
# Deep SORT = multi-object tracker
# OpenCV = video input/output and visualization
# ------------------------------------------------------------

MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "walkers.mp4"       # Change to 0 for webcam input
CONFIDENCE = 0.35
TARGET_CLASSES = {0, 2, 3, 5, 7}  # person, car, motorcycle, bus, truck

model = YOLO(MODEL_PATH)

# Explicitly use Deep SORT as required by Task 4.
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_cosine_distance=0.3,
    nn_budget=100,
    embedder="mobilenet",
)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {VIDEO_PATH}")

# Keep a short trail for every confirmed track.
trail = defaultdict(lambda: deque(maxlen=30))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. YOLO detects objects in the current frame.
    result = model(frame, conf=CONFIDENCE, verbose=False)[0]

    detections = []

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            if class_id not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # Deep SORT expects [left, top, width, height].
            detections.append(
                ([float(x1), float(y1), float(width), float(height)],
                 float(confidence),
                 model.names[class_id])
            )

    # 2. Deep SORT assigns persistent tracking IDs.
    tracks = tracker.update_tracks(detections, frame=frame)

    annotated_frame = frame.copy()

    # 3. Draw bounding boxes, labels, IDs and movement trails.
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        class_name = track.get_det_class() or "object"

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        trail[track_id].append((cx, cy))

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"{class_name} | ID: {track_id}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        cv2.circle(
            annotated_frame,
            (cx, cy),
            4,
            (0, 255, 0),
            -1,
        )

        points = list(trail[track_id])
        for i in range(1, len(points)):
            cv2.line(
                annotated_frame,
                points[i - 1],
                points[i],
                (0, 255, 255),
                2,
            )

    cv2.putText(
        annotated_frame,
        "YOLOv8 + Deep SORT | Press Q to quit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Task 4 - Object Detection and Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
