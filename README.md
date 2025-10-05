# QML-IDS: Hybrid Quantum-Classical LSTM for Network Intrusion Detection

This repository contains the official source code and a fully reproducible Docker environment for the research paper exploring a hybrid Quantum-Classical model for Network Intrusion Detection.

The project investigates the application of Quantum Machine Learning (QML) to improve the detection of malicious network traffic by augmenting a classical Long Short-Term Memory (LSTM) network with a Parameterized Quantum Circuit (PQC). Our findings show that while a classical LSTM baseline achieves a higher overall F1-Score, the hybrid QLSTM model consistently demonstrates superior **precision**, suggesting its potential for building high-fidelity IDS with fewer false alarms.

[![Docker Pulls](https://img.shields.io/docker/pulls/haodpsut/qml-ids.svg)](https://hub.docker.com/r/haodpsut/qml-ids)

## 🔬 Reproducible Research Environment

To guarantee 100% reproducibility and bypass the infamous "dependency hell," the entire experimental environment is encapsulated within a Docker image. This image contains a complete, pre-configured toolchain with GPU support for PyTorch and a stable CPU-based quantum simulation backend with PennyLane.

### Prerequisites

*   A Linux-based host machine.
*   [Docker Engine](https://docs.docker.com/engine/install/).
*   An NVIDIA GPU with compatible drivers.
*   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
*   A Kaggle account and API token (`kaggle.json`).

## 🚀 Quick Start: Full Experiment Replication

Follow these steps to set up the environment, download the data, and run all experiments.

### Step 1: Set Up the Environment

Clone the repository and pull the pre-built Docker image.

```bash
# Clone this repository
git clone https://github.com/ailabteam/qml-ids.git
cd qml-ids

# Pull the stable Docker image from Docker Hub
docker pull haodpsut/qml-ids:latest
```
*(Note: The current stable version is `1.1` or higher.)*

### Step 2: Download the Dataset

This project uses the CIC-IDS2017 dataset. We will use the Kaggle API to download it.

1.  **Get your Kaggle API Token:**
    *   Log into your Kaggle account.
    *   Go to `Settings -> API` and click `Create New Token`. This will download a `kaggle.json` file.

2.  **Place the token and download data:**
    *   Create a `./.kaggle` directory in the project root.
    *   Place your `kaggle.json` file inside it.
    *   Run the download script (this uses a temporary `tools` conda environment to avoid polluting the system):
    ```bash
    # Create a temporary environment to download data
    conda create -n tools python=3.10 kaggle -c conda-forge -y
    conda activate tools
    
    # Run download (using the dataset version that provides 8 separate files)
    kaggle datasets download -d cicids2017/cicids2017-dataset -p ./data --unzip
    
    # Move files to the root data directory
    mv ./data/MachineLearningCSV/MachineLearningCVE/*.csv ./data/
    rm -r ./data/MachineLearningCSV
    
    # Deactivate and remove the temporary environment
    conda deactivate
    conda env remove -n tools
    ```
    Your `./data` directory should now contain 8 `.csv` files.

### Step 3: Run the Experiments

Start the Docker container. This command mounts your local project directory into the container's `/app` workspace.

```bash
docker run --rm -it --gpus all -v $(pwd):/app haodpsut/qml-ids:latest
```

Once inside the container's shell (`(/opt/env) root@...:/app#`), run the scripts in order:

```bash
# 1. Exploratory Data Analysis and Preprocessing
python 01_eda.py
python 02_preprocessing.py

# 2. Train and evaluate the models
python 03_train_qlstm.py
python 04_train_lstm_baseline.py

# 3. (Optional) Run hyperparameter experiments
python 03a_train_qlstm_h64.py
python 03b_train_qlstm_q8.py

# 4. Generate the final comparison plot
python 05_plot_results.py
```
All results, including cleaned data, trained models, and figures, will be saved in the `./output` directory on your host machine.

## ❤️ The Journey: A Guide to Overcoming Installation Hell

Building this stable environment was a formidable challenge. This section documents the painful lessons learned, hoping to save others from the same struggle.

1.  **The `Illegal Instruction` Nightmare:**
    *   **Problem:** Our initial attempts using high-performance simulators (`pennylane-lightning`) consistently crashed with `Illegal instruction (core dumped)`.
    *   **Root Cause:** Pre-compiled Python wheels (`.whl`) on PyPI and even libraries from NVIDIA's Conda channel (`cuQuantum`) were built with modern CPU instruction sets (AVX/AVX2). Our server's CPU did not support them. Docker virtualizes the OS, **but not the CPU architecture**.
    *   **Lesson:** For high-performance computing, you cannot trust pre-compiled binaries blindly. We attempted to recompile `pennylane-lightning` from source, explicitly disabling AVX flags, but the dependency on the pre-compiled `cuQuantum` library made this a dead end.

2.  **The Build Dependency Maze:**
    *   **Problem:** Compiling from source failed repeatedly due to missing tools.
    *   **Root Cause:** The build process required a specific toolchain: a compatible C++ compiler (`g++ 12` was the sweet spot), `cmake`, `python-dev` headers, and the full CUDA Toolkit (`nvcc`), not just the runtime.
    *   **Lesson:** A `runtime` environment is not a `development` environment. Building complex C++/CUDA extensions requires a complete development toolchain.

3.  **The Ultimate Solution: Stability over Speed:**
    *   **Problem:** The high-performance GPU simulator (`lightning.gpu`) was unattainable due to the CPU-incompatible `cuQuantum` dependency.
    *   **Final Decision:** We pivoted to a pragmatic and robust hybrid architecture: **PyTorch on GPU** for the heavy classical computations (LSTM layers) and **PennyLane on CPU** using the universal `default.qubit` simulator. This simulator is written in Python/NumPy and has no low-level hardware dependencies, guaranteeing it will run anywhere.
    *   **Packaging with `conda-pack`:** Instead of replicating the complex installation in a Dockerfile, we created a perfect Conda environment on the host, then used `conda-pack` to archive it. The Dockerfile simply unpacks this guaranteed-to-work environment, leading to a fast, reliable, and 100% reproducible build.

This journey highlights a critical principle in computational science: **reproducibility and stability are paramount.**
