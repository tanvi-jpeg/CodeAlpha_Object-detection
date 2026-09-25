# Object Detection & Tracking

A real-time **object detection, tracking, counting, and segmentation system** built using **YOLO (Ultralytics)** and **OpenCV**.

The project can identify objects in images/videos, track detected objects across frames, count objects, and visualize detection results with bounding boxes, class labels, confidence scores, and tracking IDs.

---

## 📌 Features

* 🔍 Real-time object detection
* 🎯 Object tracking across video frames
* 🔢 Object counting
* 🧍 People detection and tracking
* 🚗 Vehicle detection
* 🛣️ Video-based detection
* 🎨 Bounding boxes and class labels
* 🆔 Unique tracking IDs
* 📈 Confidence scores
* 🎭 Object segmentation
* 📹 Support for video input
* 🖼️ Support for image input
* ⚡ Powered by YOLO and OpenCV

---

## 🛠️ Technologies Used

| Technology         | Purpose                                   |
| ------------------ | ----------------------------------------- |
| Python             | Core programming language                 |
| YOLO (Ultralytics) | Object detection, tracking & segmentation |
| OpenCV             | Video/image processing                    |
| NumPy              | Numerical operations                      |
| PyTorch            | Deep learning framework                   |

---

## 📂 Project Structure

```text
object-detection/
│
├── counting.py              # Object counting
├── tracking_ppl.py          # People tracking
├── people_with_trail.py    # People tracking with movement trails
├── segmentation.py         # Object segmentation
│
├── street.mp4              # Sample input video
├── yolov8n.pt              # YOLO model weights
│
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Git ignored files
```

> **Note:** Large model files and videos may be excluded from GitHub using `.gitignore` or Git LFS.

---

## ⚙️ Requirements

Make sure you have:

* Python 3.10+
* pip
* Git
* A computer capable of running PyTorch/YOLO

Check your Python version:

```bash
python --version
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/object-detection.git
```

Move into the project directory:

```bash
cd object-detection
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main packages manually:

```bash
pip install ultralytics opencv-python numpy
```

---

## 🧠 YOLO Model

This project uses an Ultralytics YOLO model.

For example:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
```

The `yolov8n.pt` model is a lightweight model suitable for testing and real-time applications.

You can replace it with another compatible YOLO model depending on your accuracy and performance requirements.

---

# 🔍 Object Detection

The detection system identifies objects present in an image or video.

Example:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("street.mp4", show=True)
```

Detected objects are displayed with:

* Bounding boxes
* Object class
* Confidence score

---

# 🎯 Object Tracking

The project also supports tracking objects across video frames.

Example:

```python
results = model.track(
    source="street.mp4",
    persist=True,
    show=True
)
```

Tracking assigns unique IDs to detected objects, allowing the system to follow the same object throughout the video.

Example:

```text
Person  ID: 1
Person  ID: 2
Car     ID: 3
```

---

# 🧍 People Tracking

The project includes functionality specifically designed to detect and track people.

Run:

```bash
python tracking_ppl.py
```

The system detects people and assigns tracking IDs to them.

This can be useful for:

* Crowd monitoring
* People counting
* Surveillance systems
* Traffic analysis
* Smart-city applications

---

# 🔢 Object Counting

The project can count detected objects in a video.

Run:

```bash
python counting.py
```

The system can be adapted to count:

* People
* Cars
* Bikes
* Trucks
* Buses
* Other supported YOLO classes

Example output:

```text
People: 8
Cars: 12
Buses: 2
```

---

# 🧍‍♂️ Movement Trails

The project can visualize the movement of tracked people using trails.

Run:

```bash
python people_with_trail.py
```

The trail represents the previous positions of the tracked object.

This helps visualize:

* Movement direction
* Walking paths
* Object trajectories
* Crowd movement

---

# 🎭 Object Segmentation

The project also contains segmentation functionality.

Run:

```bash
python segmentation.py
```

Unlike simple object detection, segmentation identifies the individual pixels belonging to an object.

Conceptually:

```text
Object Detection
       ↓
Bounding Box
       ↓
┌─────────────┐
│    Person   │
└─────────────┘

Segmentation
       ↓
Object Mask
       ↓
Exact pixels belonging to object
```

---

# 🎥 Input Sources

The system can work with different input sources.

### Video

```python
model.track(source="street.mp4")
```

### Webcam

```python
model.track(source=0, show=True)
```

### Image

```python
model("image.jpg", show=True)
```

---

# 📊 Supported Object Classes

The pretrained YOLO model can detect common objects such as:

* Person
* Car
* Bicycle
* Motorcycle
* Bus
* Truck
* Traffic light
* Dog
* Cat
* Bottle
* Chair
* Laptop
* Phone

The exact classes depend on the YOLO model being used.

---

# 📁 Requirements File

Create a `requirements.txt` file containing:

```text
ultralytics
opencv-python
numpy
```

PyTorch will be installed as a dependency of Ultralytics. For systems with specific CPU/GPU requirements, install the appropriate PyTorch build separately if necessary.

---

# ▶️ Running the Project

After activating the virtual environment:

### Object Counting

```bash
python counting.py
```

### People Tracking

```bash
python tracking_ppl.py
```

### People Tracking with Trails

```bash
python people_with_trail.py
```

### Segmentation

```bash
python segmentation.py
```

---

# 🖥️ Output

The application displays the processed video with:

```text
┌──────────────────────────────────────┐
│                                      │
│       PERSON                         │
│       ┌───────────┐                  │
│       │           │                  │
│       │   Object  │                  │
│       │           │                  │
│       └───────────┘                  │
│        ID: 1                         │
│        Confidence: 0.92              │
│                                      │
└──────────────────────────────────────┘
```

Depending on the script, the output may also contain object counts, tracking IDs, segmentation masks, and movement trails.

---

# 🌐 Deployment

This project is primarily a **Python-based computer vision application** and is designed to run locally.

Traditional static hosting platforms such as **GitHub Pages, Netlify, or Vercel static hosting cannot directly run the YOLO/Python inference process**.

For a web-based version, the recommended architecture is:

```text
              User
               │
               ▼
        ┌──────────────┐
        │ Web Frontend │
        │ HTML/CSS/JS  │
        └──────┬───────┘
               │
               │ API Request
               ▼
        ┌──────────────┐
        │ Python API   │
        │ Flask/FastAPI│
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ YOLO Model   │
        │ + OpenCV     │
        └──────┬───────┘
               │
               ▼
          Detection
             Result
```

The frontend can be deployed using platforms such as Vercel or Netlify, while the Python/YOLO backend should be deployed on a platform that supports Python applications and the required ML dependencies.

---

# ⚠️ Performance Considerations

YOLO inference can require significant CPU/GPU resources.

For better performance:

* Use a smaller YOLO model for CPU systems.
* Use GPU acceleration when available.
* Resize large input videos.
* Process only required object classes.
* Avoid storing unnecessarily large video/model files in Git.
* Use an appropriate PyTorch build for the target hardware.

---

# 🔐 Important Notes

The pretrained model weights and sample videos can be large.

Avoid committing unnecessary large files directly to GitHub.

Use `.gitignore` for files such as:

```text
.venv/
__pycache__/
*.pyc
.env
runs/
*.mp4
*.avi
```

If model weights need to be version-controlled, consider **Git LFS**.

---

# 🧪 Future Improvements

Possible improvements include:

* 🌐 Web interface
* 📤 Image/video upload
* 📷 Live webcam detection
* 📊 Real-time analytics dashboard
* 🔢 Advanced object counting
* 🚦 Traffic monitoring
* 🧍 Crowd density estimation
* 🚨 Restricted-zone detection
* 🔔 Alert/notification system
* ☁️ Cloud deployment
* 📱 Mobile-friendly interface
* 🤖 Custom-trained YOLO model

---

# 🎯 Applications

This project can be adapted for:

* Smart surveillance
* Traffic monitoring
* Crowd monitoring
* Vehicle counting
* Pedestrian detection
* Retail analytics
* Smart-city systems
* Industrial monitoring
* Security systems
* Computer vision research

---

# 👩‍💻 Author

**Tanvi**

Computer Science & Engineering Student

GitHub:
https://github.com/tanvi-jpeg

---

# ⭐ Acknowledgements

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* [OpenCV](https://opencv.org/)
* [PyTorch](https://pytorch.org/)

---

## 📄 License

This project is intended for educational and academic purposes.
