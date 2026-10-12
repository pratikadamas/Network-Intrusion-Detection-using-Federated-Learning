# Network Intrusion Detection using Federated Learning (FL-NID)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-FF6F00?logo=flower&logoColor=white)](https://flower.ai/)
[![Dataset](https://img.shields.io/badge/Dataset-NF--ToN--IoT--v2-00599C)](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
[![GPU](https://img.shields.io/badge/GPU%20Acceleration-NVIDIA%20CUDA-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)

A privacy-preserving, decentralized Network Intrusion Detection System (NIDS) designed for Internet-of-Things (IoT) environments using **Federated Learning (Flower)** and **Deep Learning (PyTorch)**. 

Trained on the large-scale **NF-ToN-IoT-v2** dataset (nearly **17 million flow records**), this project eliminates data privacy leakage by training local intrusion detection models at the edge while sharing only model weights with a central aggregator.

---

## 📌 Table of Contents
- [Architecture Overview](#-architecture-overview)
- [Dataset (NF-ToN-IoT-v2)](#-dataset-nf-ton-iot-v2)
- [Data Preprocessing & Engineering](#-data-preprocessing--engineering)
- [Repository Structure](#-repository-structure)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Workflow & Execution](#-workflow--execution)
- [Federated Learning Simulation](#-federated-learning-simulation)
- [License](#-license)

---

## 🏗 Architecture Overview

Traditional centralized intrusion detection requires aggregating raw, high-volume network telemetry onto a single central server, creating bandwidth bottlenecks, single points of failure, and major privacy/compliance concerns.

```
                  +-----------------------------------+
                  |      Central Aggregator (Server)  |
                  |  - Global Model (FedAvg / FedProx)|
                  |  - Evaluates on global_test set   |
                  +-----------------------------------+
                         ▲        │        ▲
      Local Model Weights│        │Weights │Local Model Weights
                         │        ▼        │
           ┌─────────────┴──┐ ┌───┴─────────────┐ ┌───┴─────────────┐
           │  Edge Client 1 │ │  Edge Client 2  │ │  Edge Client K  │
           │  (IoT Subnet A)│ │  (IoT Subnet B) │ │  (IoT Subnet K) │
           │  Private Data  │ │  Private Data   │ │  Private Data   │
           └────────────────┘ └─────────────────┘ └─────────────────┘
```

1. **Decentralized Local Training:** IoT edge nodes/gateways train deep neural networks on their local flow records.
2. **Federated Aggregation:** Clients submit only encrypted parameter updates/gradients.
3. **No Raw Data Sharing:** Raw payload data and IP communication logs never leave the client's local perimeter.

---

## 📊 Dataset (NF-ToN-IoT-v2)

This project utilizes **NF-ToN-IoT-v2**, a NetFlow-v2 formatted network intrusion benchmark generated from the IoT testbed created by the University of New South Wales (UNSW) Canberra:

* **Total Records:** **16,940,497 network flows** (~2.68 GB on disk).
* **Feature Schema:** 43 standardized NetFlow-v2 metrics + 2 target attributes:
  * `Label`: Binary classification (`0` = Benign, `1` = Attack).
  * `Attack`: Multi-class category (`Benign`, `ddos`, `dos`, `reconnaissance/scanning`, `ransomware`, `backdoor`, `injection`, `password`, `xss`, `mitm`).

---

## ⚙️ Data Preprocessing & Engineering

Due to the massive 17M row scale and memory-constrained environments, data preparation is executed using a **memory-safe chunked streaming engine** (`chunksize=250,000`), keeping RAM consumption constant at **under 50 MB**:

### 1. Identifier Pruning (Preventing Shortcut Learning)
* **Dropped:** `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, and `DNS_QUERY_ID`.
* **Rationale:** Feeding raw IP addresses or ephemeral random ports causes deep neural networks to memorize specific host subnets rather than learning generalized attack traffic behaviors.

### 2. Sanitization
* Sub-millisecond flows (`FLOW_DURATION = 0`) induce division-by-zero errors in flow rates and throughput metrics.
* Vectorized replacement with `np.nan_to_num()` ensures all `NaN`, `+Inf`, and `-Inf` artifacts are mapped to `0.0`.

### 3. Log-Transformation (`log1p`)
* Network metrics (`IN_BYTES`, `OUT_BYTES`, throughput) exhibit extreme power-law heavy-tailed distributions ($0 \to 10^8+$).
* Applying $\text{log1p}(x) = \ln(1 + x)$ smoothly compresses dynamic ranges to $[0, \approx 18.42]$ while preserving natural zeros.

### 4. Zero-Drift Global Z-Score Standardization
* To prevent **client drift** in federated settings, scaling parameters ($\mu, \sigma$) are calibrated on a reference baseline and saved to `global_scaler.joblib`.
* Standardized features are clamped to $[-5.0, 5.0]$ for PyTorch gradient stability, stored at `float32` precision.

---

## 📂 Repository Structure

```
FL-NID/
├── NF-ToN-IoT-v2/               # [Git-ignored] Raw and preprocessed dataset directory
│   └── data/
│       ├── NF-ToN-IoT-v2.csv          # Raw NetFlow dataset (~2.68 GB)
│       ├── NF-ToN-IoT-v2_pruned.csv   # Pruned & sanitized dataset (~2.08 GB)
│       ├── NF-ToN-IoT-v2_normalized.csv # Fully normalized training dataset
│       └── global_scaler.joblib       # Serialized StandardScaler
│
├── EDA.ipynb                    # End-to-end Chunked EDA & Normalization Pipeline
├── requirements.txt             # Python project dependencies
├── .gitignore                   # Excludes raw data files and temporary scripts
└── README.md                    # Project documentation
```

---

## 🚀 Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/pratikadamas/Network-Intrusion-Detection-using-Federated-Learning.git
cd Network-Intrusion-Detection-using-Federated-Learning
```

### 2. Create and Activate Virtual Environment
```bash
# Using Conda
conda create -n DL-FL python=3.11 -y
conda activate DL-FL

# Or using venv
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate # On Linux/macOS
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. GPU Verification
Verify that PyTorch detects your CUDA-enabled GPU (e.g., RTX 2050):
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

---

## 📓 Workflow & Execution

1. **Exploratory Data Analysis & Normalization:**
   * Open [`EDA.ipynb`](file:///e:/FL-NID/EDA.ipynb) in Jupyter / VSCode.
   * Run cells sequentially to stream `NF-ToN-IoT-v2.csv` in chunks.
   * Generates the clean, normalized, ready-to-train file: `NF-ToN-IoT-v2_normalized.csv`.

2. **Model Training & FL Simulation:**
   * Uses the pre-scaled `NF-ToN-IoT-v2_normalized.csv` directly without any runtime scaling overhead.
   * Partitions data across simulated edge IoT devices to evaluate non-IID performance under **FedAvg** and **FedProx**.

---

## 🛡️ License

This project is licensed under the MIT License - see the LICENSE file for details.
