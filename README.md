---

# 🚗 Lane Detection using Computer Vision (OpenCV)

This project implements a **lane detection pipeline** using traditional **computer vision techniques** in Python.
It processes road videos to detect **lane lines** (white and yellow) using **Canny edge detection, region masking, and Hough Line Transform**.

---

## 📌 Features

* Detects **left and right lane lines** from road videos
* Uses classical image processing (no deep learning)
* Works on both **white lane** and **yellow lane** videos
* Smooth lane estimation using **slope-weighted averaging**
* Outputs processed videos with detected lanes overlaid

---

## 🧠 Pipeline Overview

1. **Grayscale Conversion**
2. **Gaussian Blur** (noise reduction)
3. **Canny Edge Detection**
4. **Region of Interest Masking**
5. **Hough Line Transform**
6. **Lane Line Averaging**
7. **Overlay on Original Frame**

---

## 🛠️ Tech Stack

* **Python 3**
* **OpenCV**
* **NumPy**
* **Matplotlib**
* **MoviePy**

---

## 📂 Project Structure

```
lane-detection/
│
├── lane_detection.py
├── test_videos/
│   ├── solidWhiteRight.mp4
│   └── solidYellowLeft.mp4
│
├── test_videos_output/
│   ├── solidWhiteRight.mp4
│   └── solidYellowLeft.mp4
│
└── README.md
```

---

## 🔧 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/lane-detection.git
cd lane-detection
```

### 2️⃣ Create virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 `requirements.txt`

```txt
numpy
opencv-python
matplotlib
moviepy
```

---

## ▶️ Usage

Update the video paths inside `lane_detection.py`, then run:

```bash
python lane_detection.py
```

Output videos will be saved in:

```
test_videos_output/
```

---

## 🎥 Sample Output

| Input Video | Output Video         |
| ----------- | -------------------- |
| White Lane  | White lane detected  |
| Yellow Lane | Yellow lane detected |


## 🎥 Output Demo

### White Lane Detection
![White Lane Detection](gifs/white_lane.gif)

### Yellow Lane Detection
![Yellow Lane Detection](gifs/yellow_lane.gif)


---

## ⚠️ Limitations

* Works best on **straight or mildly curved roads**
* Sensitive to lighting changes
* Not robust for heavy shadows or complex road markings
* Classical CV approach (no ML/DL)

---

## 🚀 Future Improvements

* Detect **center lane only** (airport runway / docking use-case)
* Color-based filtering for **yellow center lines**
* Curved lane detection using polynomial fitting
* Upgrade to **YOLO / segmentation-based approach**
* Real-time inference support

---

## 👨‍💻 Author

**Sreenath**
Computer Vision Engineer
Experience in OCR, Object Detection, Robotics & Autonomous Systems

---

## 📜 License

This project is licensed under the **MIT License**.

---


