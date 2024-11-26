# 🏖️ Beach Crowd Detection System

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A computer vision project that implements multiple techniques to detect and count people in beach surveillance images. This system provides various methods for crowd analysis and visualization tools for comparing different detection approaches.

## 📋 Table of Contents
- [Overview](#overview)
- [Methods Implemented](#methods-implemented)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results Visualization](#results-visualization)
- [Parameter Tuning](#parameter-tuning)
- [Contributing](#contributing)

## 🔍 Overview
This project focuses on analyzing beach crowd patterns using computer vision techniques. It implements multiple detection methods and provides comprehensive visualization tools for analyzing the results. The system is designed to handle various beach conditions and crowd densities.

## 🛠️ Methods Implemented

### 1. Background Subtraction Method
- Median background modeling
- Adaptive thresholding
- Morphological operations for noise reduction
- Connected component analysis

### 2. Edge Detection with CLAHE
- Contrast Limited Adaptive Histogram Equalization
- Canny edge detection
- Contour analysis
- Size-based filtering

### 3. Adaptive Thresholding Method
- Gaussian adaptive thresholding
- Morphological processing
- Component labeling
- Area-based filtering

## 📁 Project Structure
```
beach-crowd-detection/
│
├── crowd_detection_background.py   # Background subtraction implementation
├── crowd_detection_edge.py         # Edge detection implementation
├── requirements.txt                # Project dependencies
├── Dataset/                        # Input images folder (not included)
│   └── *.jpg                      # Beach surveillance images
│
├── detection_results/              # Output from background subtraction
│   ├── presentation_visuals/       # Visualization images
│   └── detection_summary.txt       # Detection results
│
└── alternative_detection_results/  # Output from edge detection
    ├── presentation_visuals/       # Visualization images
    └── detection_summary.txt       # Detection results
```

## ⚙️ Requirements
- Python 3.7+
- OpenCV (cv2)
- NumPy
- SciPy
- Matplotlib

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/beach-crowd-detection.git
cd beach-crowd-detection
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Prepare your dataset:
   - Create a `Dataset` folder in the project root
   - Add your beach surveillance images to this folder

2. Run background subtraction method:
```bash
python crowd_detection_background.py
```

3. Run edge detection method:
```bash
python crowd_detection_edge.py
```

## 📊 Results Visualization

The system generates comprehensive visualizations including:

- **Detection Process Visualization**
  - Original image
  - Background model
  - Detection mask
  - Final detection overlay

- **Analysis Graphs**
  - People count trends
  - Method comparison charts
  - Detection accuracy visualization

## 🎯 Parameter Tuning

Key parameters that can be adjusted for better detection:

```python
# Background Subtraction Parameters
min_person_size = 25    # Minimum size for detection
threshold_value = 35    # Binary threshold value
kernel_size = (3,3)     # Morphological operation kernel
morph_iterations = 2    # Number of morphological iterations

# Edge Detection Parameters
clip_limit = 2.0        # CLAHE clip limit
tile_size = (8,8)       # CLAHE tile size
canny_thresh = (50,150) # Canny edge detection thresholds
```

## 🔧 Image Processing Pipeline

### Background Subtraction Method
1. Convert images to grayscale
2. Create background model using median averaging
3. Perform background subtraction
4. Apply Gaussian blur for noise reduction
5. Apply binary thresholding
6. Use morphological operations for cleanup
7. Detect and count connected components

### Edge Detection Method
1. Apply CLAHE for contrast enhancement
2. Convert to grayscale
3. Apply Gaussian blur
4. Detect edges using Canny detector
5. Apply morphological operations
6. Find and filter contours
7. Count detected objects

## 📈 Performance Optimization

The detection accuracy can be improved by adjusting these factors:

- **Lighting Conditions**
  - Adjust CLAHE parameters for different times of day
  - Modify threshold values for varying light conditions

- **Distance Considerations**
  - Adjust size filters based on camera position
  - Modify detection parameters for different viewing angles

- **Crowd Density**
  - Fine-tune morphological operations for different crowd levels
  - Adjust connection parameters for grouped people

