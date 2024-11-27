# 🌊 Beach Crowd Monitoring System

<div align="center">

![Beach Monitoring](https://img.shields.io/badge/🏖️-Beach%20Monitoring-blue)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
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
- [Configuration](#-configuration)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

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
│   ├── config/                 # Configuration files
│   │   ├── config.yaml        # System parameters
│   │   └── default_params.py  # Parameter handling
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
- PyYAML

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

### Configuration

Modify `config/config.yaml` to adjust:
- Detection parameters
- Background modeling settings
- Evaluation metrics
- Visualization options

---

## 🔧 Technical Details

### Background Modeling
- Statistical modeling using multiple techniques
- Region-specific processing for water and sand areas
- Robust to lighting changes and environmental factors

### People Detection
- Multi-scale detection for varying distances
- Adaptive thresholding based on image regions
- Shape and size-based filtering

### Evaluation System
- Ground truth comparison
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

## 📈 Future Improvements

- [ ] Deep learning integration
- [ ] Real-time processing optimization
- [ ] Weather condition adaptation
- [ ] Mobile deployment
- [ ] Time-series analysis
- [ ] Crowd density estimation
- [ ] Automated alert system
- [ ] API development

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Contact

Joyarup Mitra - [@joyarup](https://github.com/joyarup) - your.email@example.com

Project Link: [https://github.com/joyarup/Beach-Crowd-Counting](https://github.com/joyarup/Beach-Crowd-Counting)

---

<div align="center">
<p>Built with ❤️ by Joyarup Mitra</p>

<p>
If you find this project useful, please consider giving it a ⭐!
</p>
</div>

