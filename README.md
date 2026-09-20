# 🕵️‍♂️ Deepfake Detection: Architectural Benchmark & Explainable AI (XAI)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

## 📌 Project Overview
This repository provides a comprehensive benchmarking pipeline for Deepfake detection, specifically evaluating the trade-offs between high-accuracy heavyweight models and efficient lightweight models suitable for edge deployment. 

The project strictly follows the extraction and evaluation protocol established in the official **FaceForensics++ (Rössler et al., ICCV 2019)** paper, ensuring academic rigor and reproducible results. Furthermore, it integrates **Explainable AI (XAI)** using Grad-CAM to visually interpret the neural networks' decision-making processes.

### Key Highlights for MLOps & FinOps:
* **Model Benchmarking:** Comparative analysis of 4 CNN architectures (Xception, ResNet50, MobileNetV2, and MesoNet-4).
* **Efficiency vs. Accuracy:** Evaluates inference speed (ms/frame) alongside AUC-ROC to optimize cloud compute costs.
* **Explainability (XAI):** Features automated Grad-CAM/Score-CAM overlays to ensure models focus on facial artifacts, preventing shortcut learning.
* **Production-Grade Pipeline:** Modularized preprocessing, dynamic input sizing, and automated metric visualizations.

---

## 📂 Repository Structure

```text
Portofolio-Deepfake-Detection/
├── assets/                     # Auto-generated benchmark metrics & XAI visualizations
├── src/                        # Core logic and execution scripts
│   ├── preprocess.py           # FF++ protocol face extraction & uniform sampling
│   ├── train_xception.py       # Heavyweight model training
│   ├── train_resnet.py         # Standard ResNet50 training
│   ├── train_mobilenet.py      # Edge-optimized lightweight model
│   ├── train_mesonet.py        # Custom Deepfake-specific architecture
│   ├── evaluate_model.py       # Automated metric computation & graphing
│   └── visualization_gradcam.py  # XAI Heatmap generation
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation


