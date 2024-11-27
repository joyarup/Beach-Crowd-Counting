# 🌊 Beach Crowd Monitoring System

<div align="center">

![Beach Monitoring](https://img.shields.io/badge/🏖️-Beach%20Monitoring-blue)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/joyarup/Beach-Crowd-Counting/graphs/commit-activity)

<p align="center">
An advanced computer vision system for real-time crowd monitoring and analysis in beach environments, combining robust background modeling with multi-scale detection techniques.
</p>

</div>

---

## 📑 Table of Contents
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technical Details](#-technical-details)
- [Results](#-results)
- [Contributing](#-contributing)
- [Future Improvements](#-future-improvements)
- [Contact](#-contact)

---

## ✨ Features

### Core Capabilities
- 🎯 Multi-scale people detection
- 🖼️ Robust background modeling
- 🌊 Region-specific processing (water/sand)
- 📊 Comprehensive evaluation metrics
- 🎨 Advanced visualization tools

### Technical Highlights
- **Background Modeling**:
  - Statistical modeling
  - Adaptive thresholding
  - Region-specific processing
- **Detection System**:
  - Multi-scale detection
  - Aspect ratio analysis
  - Shape-based filtering
- **Evaluation Metrics**:
  - Precision and recall
  - Mean Square Error (MSE)
  - Ground truth comparison

---

## 🏗️ System Architecture

```
beach_monitoring/
├── src/
│   ├── utils/                 # Utility functions
│   │   └── image_loader.py    # Image processing utilities
│   ├── detection/             # Core detection modules
│   │   ├── background.py      # Background modeling
│   │   └── people_detector.py # People detection algorithms
│   ├── visualization/         # Visualization tools
│   │   └── visualizer.py      # Result visualization
│   ├── evaluation/            # Evaluation tools
│   │   └── metrics.py         # Performance metrics
│   └── main.py               # Main execution script
└── requirements.txt          # Project dependencies
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- OpenCV
- NumPy
- SciPy
- Matplotlib
- Pandas

### Setup Steps

1. **Clone the Repository**
```bash
git clone https://github.com/joyarup/Beach-Crowd-Counting.git
cd Beach-Crowd-Counting
```

2. **Create Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🎮 Usage

### Basic Usage

1. **Prepare Your Data**
   - Place images in `Dataset` folder
   - Add ground truth CSV if available

2. **Run the System**
```bash
cd src
python main.py
```

---

## 🔧 Technical Details

### Background Modeling
- Statistical modeling using multiple techniques
- Region-specific processing for water and sand areas
- Robust to lighting changes and environmental factors

### People Detection System
The system employs a sophisticated multi-scale detection approach optimized for beach environments:

#### Multi-Scale Processing
- Uses three different scales (1.0, 0.75, 0.5) to detect people at varying distances
- Adapts detection parameters based on scale to handle perspective changes
- Scale-specific size filtering for accurate detection at different distances

#### Region-Specific Processing
- Separate processing for upper (distant) and lower (near) regions
- Adaptive thresholding with different sensitivities:
  - More sensitive parameters for distant objects (upper region)
  - Less sensitive parameters for near objects (lower region)
- Ignores detections in the top 40% of the image to reduce false positives

#### Detection Pipeline
1. **Pre-processing**:
   - Grayscale conversion
   - Background subtraction
   - Gaussian blur for noise reduction

2. **Contour Detection**:
   - Scale-dependent morphological operations
   - Dynamic kernel size adjustment based on scale
   - Adaptive thresholding for different image regions

3. **Filtering Mechanism**:
   - Area-based filtering with scale-adjusted thresholds
   - Aspect ratio analysis (0.2-4.0 for distant objects, 0.3-3.0 for near objects)
   - Solidity threshold adaptation (0.2 for distant, 0.3 for near objects)
   - Convex hull analysis for shape verification

4. **Post-processing**:
   - Contour scaling and normalization
   - Final mask generation
   - Morphological cleanup operations

### Evaluation System
- Ground truth comparison capability
- Multiple performance metrics
- Per-image and overall statistics

---

## 📊 Results

The system outputs:
- Detection visualizations
- Performance metrics
- Background models
- Evaluation reports

Example metrics include:
- Average precision and recall
- Mean Square Error (MSE)
- False positive/negative rates

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👥 Contact

Arup Sarkar- [@joyarup](https://github.com/joyarup) - arupbks@gmail.com

Project Link: [https://github.com/joyarup/Beach-Crowd-Counting](https://github.com/joyarup/Beach-Crowd-Counting)

---

<div align="center">
<p>Built with ❤️ by Arup Sarkar</p>

<p>
If you find this project useful, please consider giving it a ⭐!
</p>
</div>
