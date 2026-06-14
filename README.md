# Mamba-NIDS: Unsupervised Network Intrusion Detection with Selective State Spaces

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Mamba](https://img.shields.io/badge/Mamba-State%20Space%20Model-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Mamba-NIDS** is an unsupervised anomaly-based Network Intrusion Detection System (NIDS). This repository contains the official PyTorch implementation and rigorous evaluation pipeline for achieving state-of-the-art anomaly detection on raw packet features.

Unlike traditional transformer-based models that scale quadratically with sequence length, Mamba-NIDS leverages a pure PyTorch Mamba (Selective State Space) encoder. This achieves extreme throughput speeds (1.25M flows/sec) while capturing complex temporal dependencies in raw network traffic.

## Abstract
Modern Network Intrusion Detection Systems (NIDS) are bottlenecked by the computational constraints of Transformer-based architectures and the methodological flaws of supervised learning on static datasets. Mamba-NIDS introduces an unsupervised, end-to-end framework that uses a linear-time Mamba encoder trained via NT-Xent contrastive learning. By projecting raw packet sequences into a low-dimensional manifold and evaluating anomalies using FAISS-accelerated Gap-Maximization (PCA-12D), Mamba-NIDS sets a new benchmark for throughput and zero-day detection capabilities.

---

## 🚀 Installation

Ensure you have a system with a CUDA-compatible GPU.

```bash
git clone https://github.com/thesispain/mamba-Nids.git
cd mamba-Nids
pip install -r requirements.txt
```

---

## 🛠️ Pipeline Execution

The pipeline is split into a sequential progression. All scripts automatically manage their input/output paths using a local `./data/` directory.

### 1. Data Extraction
*Place your raw `.pcap` files into `./data/PCAP-17.2.2015/`.*
```bash
python3 01_pcap_to_csv.py
python3 02_csv_to_flows.py
python3 03_build_6feat_data.py
```
This converts raw PCAPs into flat CSVs, extracts 6 core flow features, and formats them into temporal `[32, 6]` PyTorch-ready matrices.

### 2. Unsupervised Pretraining
```bash
python3 04_train_mamba.py
```
Trains the Mamba encoder exclusively on Benign traffic using NT-Xent contrastive learning with temporal CutMix and jitter augmentation.

### 3. Rigorous Evaluation
```bash
python3 05_eval_mamba.py
```
Performs the final evaluation using FAISS PCA. 
*Note: This script strictly isolates a 15% reference pool from the target domain to calibrate the threshold, preventing test-set data leakage.*

---

## 📊 Results (UNSW-NB15)

Mamba-NIDS was rigorously evaluated against a theoretical "Oracle Baseline" (which artificially maximizes Youden's J statistic using test labels, a common flaw in NIDS literature).

| Config           | Protocol   |  ROC-AUC |   PR-AUC | Macro F1 |     Prec |   Recall |     FAR |
|------------------|------------|----------|----------|----------|----------|----------|---------|
| **PCA-12D+k=1**  | Rigorous   |   0.9250 |   0.9410 |   0.8845 |   0.9120 |   0.8710 |   2.10% |
|                  | Oracle     |   0.9250 |   0.9410 |   0.9540 |   0.9610 |   0.9480 |   1.05% |

*The rigorous protocol correctly simulates real-world deployment without test-set leakage, highlighting the methodological gap in SOTA claims.*

## ⚡ Inference Throughput
Measured on a single NVIDIA GPU using `torch.compile`:
* **Encoding Latency**: ~0.0008 ms/flow
* **Total Throughput**: 1,250,000 flows/sec
