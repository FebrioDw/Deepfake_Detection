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
│   ├── evaluasi_model.py       # Automated metric computation & graphing
│   └── visualisasi_gradcam.py  # XAI Heatmap generation
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📊 Benchmark Results

*(The table below illustrates the performance comparison across the architectures based on the test set.)*

| Architecture | Accuracy | F1-Score | AUC-ROC | Inference Speed | Parameters | Best Use Case |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Xception** | **0.9621** | **0.9618** | **0.9901** | ~0.00 ms/frame | 20.8 M | Cloud-based Batch Processing |
| **ResNet50** | 0.9029 | 0.9057 | 0.9707 | ~0.00 ms/frame | 23.5 M | Standard GPU Inference |
| **MobileNetV2**| 0.9129 | 0.9173 | 0.9841 | ~0.00 ms/frame | 2.2 M | Mobile / Edge Devices (FinOps) |
| **MesoNet-4** | 0.8321 | 0.8471 | 0.9205 | ~0.00 ms/frame | 0.02 M | Specialized Rapid Filtering |

> **Note:** *Inference speed metrics will be updated after running the hardware warm-up via `evaluasi_model.py`.*

### 🗂️ Dataset Scope & Limitation
Please note that all models in this repository were trained, validated, and evaluated **exclusively on the FaceForensics++ dataset**. While the benchmark results demonstrate high accuracy within this specific distribution (e.g., Xception achieving 96% accuracy), the models may require further fine-tuning or cross-dataset training to generalize effectively against novel manipulation techniques or "in-the-wild" deepfakes outside the FaceForensics++ domain.

### Performance Visualization
<!-- 
Catatan: Hapus tanda panah komentar ini nanti jika gambar grafik sudah Anda masukkan ke folder assets/
![Benchmark Metrics](assets/benchmark_metrics.png)
![ROC Curves](assets/roc_curves.png)
-->
*(Performance graphs and ROC curves will be attached here shortly after the automated evaluation pipeline is fully executed).*

---

## 🧠 Explainable AI (Grad-CAM)

To ensure the models are detecting actual facial manipulation artifacts rather than background noise, Grad-CAM (Gradient-weighted Class Activation Mapping) is applied to the final convolutional layers.

<!-- 
Catatan: Hapus tanda panah komentar ini nanti jika gambar wajah heatmap sudah Anda masukkan ke folder assets/
![Grad-CAM Comparison](assets/xai_gradcam_comparison.png)
-->
*(Grad-CAM visual overlays will be uploaded soon. Warmer colors will indicate the exact pixel regions the neural network heavily weighted to classify the image as a Deepfake).*

## 🚀 Quick Start Guide

### 1. Environment Setup
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/FebrioDw/Deepfake_Detection.git](https://github.com/FebrioDw/Deepfake_Detection.git)
cd Deepfake_Detection
pip install -r requirements.txt
```

### 2. Data Preprocessing
Ensure your dataset is placed in the `dataset/` folder, then run the FF++ extraction protocol:
```bash
python src/preprocess.py
```

### 3. Model Training
Train any of the desired architectures. The scripts automatically handle dynamic input sizing and checkpointing:
```bash
python src/train_xception.py
python src/train_mobilenet.py
```

### 4. Evaluation & Benchmarking
Generate the metrics table, confusion matrices, and ROC curves:
```bash
python src/evaluasi_model.py
```

### 5. XAI Interpretability
Generate the Grad-CAM heatmaps for manipulated frames:
```bash
python src/visualisasi_gradcam.py
```

---
*Designed and engineered for robust, explainable, and cost-efficient Deepfake detection.*

