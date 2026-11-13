# 🕵️‍♂️ Deepfake Detection: Architectural Benchmark & Explainable AI (XAI)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![React](https://img.shields.io/badge/React-Vite-blueviolet)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

## 📌 Project Overview
This repository provides a comprehensive full-stack pipeline for Deepfake detection. It bridges academic research with software engineering by evaluating the trade-offs between high-accuracy heavyweight models and efficient lightweight models, while deploying them through a modern web application interface.

The core machine learning pipeline strictly follows the extraction and evaluation protocol established in the official **FaceForensics++ (Rössler et al., ICCV 2019)** paper. Furthermore, it integrates **Explainable AI (XAI)** using Grad-CAM to visually interpret the neural networks' decision-making processes, ensuring the models focus on facial artifacts rather than background noise.

### Key Highlights for MLOps & Software Engineering:
- **Model Benchmarking:** Comparative analysis of 4 CNN architectures (Xception, ResNet50, MobileNetV2, and MesoNet-4).
- **Full-Stack Deployment:** Features a decoupled architecture with a modern React (Vite) frontend for user interaction and a robust FastAPI backend for asynchronous model inference.
- **Explainability (XAI):** Features automated Grad-CAM/Score-CAM overlays to prevent shortcut learning.
- **Production-Grade Engineering:** Modularized preprocessing, dynamic input sizing, and scalable API endpoints.

---

## 📂 Repository Structure

The project is structured into three main modules: Model Training, Web Backend, and Web Frontend.

```text
Deepfake_Detection/
├── Training/                   # Core ML Logic & Experimentation
│   ├── src/                    # Training, evaluation, and XAI scripts
│   └── requirements.txt        # ML dependencies
├── Web/                        # Full-Stack Application
│   ├── backend/                # FastAPI Application (Inference Engine)
│   │   ├── detector.py         # Model loading and prediction logic
│   │   ├── main.py             # API Endpoints
│   │   └── requirements.txt    # Backend dependencies
│   └── frontend/               # React / Vite User Interface
│       ├── public/             # Static web assets
│       ├── src/                # UI Components and state management
│       ├── package.json        # Node.js dependencies
│       └── vite.config.js      # Bundler configuration
├── models/                     # (Ignored in Git) Pre-trained .keras weights
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

### 🗂️ Dataset Scope & Limitation
Please note that all models in this repository were trained, validated, and evaluated **exclusively on the FaceForensics++ dataset**. While the benchmark results demonstrate high accuracy within this specific distribution, the models may require further cross-dataset training to generalize effectively against novel manipulation techniques in the wild.

---

## 🧠 Explainable AI (Grad-CAM)
To ensure the models are detecting actual facial manipulation artifacts rather than background noise, Grad-CAM (Gradient-weighted Class Activation Mapping) is applied to the final convolutional layers. Warmer colors indicate the exact pixel regions the neural network heavily weighted to classify the image as a Deepfake.

---

## 🚀 Quick Start Guide (Web Application)

To run the Deepfake Detection web application locally, you will need to start both the Backend and Frontend servers.

### 1. Start the FastAPI Backend
Navigate to the backend directory, install the Python requirements, and launch the Uvicorn server:
```bash
cd Web/backend
pip install -r requirements.txt
uvicorn main:app --reload
```
*The backend API will run on `http://localhost:8000`*

### 2. Start the React Frontend
Open a new terminal, navigate to the frontend directory, install the Node modules, and start the Vite development server:
```bash
cd Web/frontend
npm install
npm run dev
```
*The web interface will typically be available at `http://localhost:5173`*

### Note on Large Model Weights
Due to GitHub file size limits, the pre-trained model weights (`.keras` files) located in the `models/` directory are excluded from this repository via `.gitignore`. You must train the models locally using the scripts in the `Training/` directory or download the weights separately before running inference through the web app.

---
*Designed and engineered for robust, explainable, and scalable Deepfake detection.*