# Applied Machine Learning
### 🎓 Technical University of Denmark

<div align="center">

<img src="https://unepccc.org/wp-content/uploads/2021/05/dtu-logo.png" alt="DTU Logo" width="120"/>

# Applied Machine Learning
### 🎓 DTU Engineering Technology

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![DTU EngTech](https://img.shields.io/badge/DTU-Engineering_Technology-F2D04B?style=for-the-badge&labelColor=F2D04B&color=F2D04B)](https://www.engtech.dtu.dk/)

<p align="center">
  <b>Slides</b> live on DTU Learn • <b>Notebooks</b> live in this repo • <b>Code</b> drives the logic
</p>

</div>

---

## 📖 Overview

Welcome to the **Applied Machine Learning** course repository. This collection is designed to provide a hands-on approach to modern ML techniques.

The material is structured so that each lecture topic corresponds to a specific Jupyter notebook, guiding you from raw data to model evaluation.

### 📂 Repository Structure

```text
.
├── 📓 notebooks/       # Interactive Lecture notebooks
├── 🛠 src/aml_course/  # Helper utilities & source code
├── 🚀 scripts/         # Standalone training/utility scripts
├── 🧠 models/          # Pretrained artifacts & weights
├── 💾 data/            # Datasets (auto-downloaded or manual)
├── 🖼 pictures/        # Generated figures & assets
└── 📝 docs/            # Extra notes & documentation

```

---

## ⚡ Getting Started

### 1️⃣ Create an Environment

Choose your preferred package manager to set up the environment.

<details open>
<summary><b>Option A: pip (Recommended for simplicity)</b></summary>

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# macOS / Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

</details>

<details>
<summary><b>Option B: conda</b></summary>

```bash
conda env create -f environment.yml
conda activate aml-course

```

</details>

> **💡 Note for Power Users:**
> For advanced sessions requiring specific GPU support, install the optional dependencies:
> `pip install -r requirements-optional.txt`

### 2️⃣ Launch Jupyter

Start the lab server:

```bash
jupyter lab

```

Navigate to the `notebooks/` folder to begin. Each notebook contains a **Setup cell** that automatically configures paths (`DATA_DIR`, `FIGURES_DIR`, etc.) relative to the repository root.

---

## 🗓️ Lecture Map

| # | Topic | 📑 Slides | 💻 Notebook |
| --- | --- | --- | --- |
| **Pre A** | 🐍 Python + NumPy Crash Course | — | [`00_python_numpy_crash_course.ipynb`](https://www.google.com/search?q=notebooks/00_prerequisites/00_python_numpy_crash_course.ipynb) |
| **Pre B** | 🐼 Pandas Essentials | — | [`01_pandas_intro.ipynb`](https://www.google.com/search?q=notebooks/00_prerequisites/01_pandas_intro.ipynb) |
| **01** | **Introduction** | [PDF] | [`01_introduction.ipynb`](https://www.google.com/search?q=notebooks/01_introduction.ipynb) |
| **02** | **ML Foundations** | [PDF] | [`02_ml_foundations.ipynb`](https://www.google.com/search?q=notebooks/02_ml_foundations.ipynb) |
| **03** | **Regression** | [PDF] | [`03_regression.ipynb`](https://www.google.com/search?q=notebooks/03_regression.ipynb) |
| **04** | **Classification** | [PDF] | [`04_classification.ipynb`](https://www.google.com/search?q=notebooks/04_classification.ipynb) |
| **05** | **Clustering** | [PDF] | [`05_clustering.ipynb`](https://www.google.com/search?q=notebooks/05_clustering.ipynb) |
| **06** | **Neural Networks** | [PDF] | [`06_neural_networks.ipynb`](https://www.google.com/search?q=notebooks/06_neural_networks.ipynb) |
| **07** | **CNNs (PyTorch)** | [PDF] | [`07_cnn.ipynb`](https://www.google.com/search?q=notebooks/07_cnn.ipynb) |
| **08** | **Resampling** | [PDF] | [`08_resampling.ipynb`](https://www.google.com/search?q=notebooks/08_resampling.ipynb) |
| **09** | **SVM** | [PDF] | [`09_svm.ipynb`](https://www.google.com/search?q=notebooks/09_svm.ipynb) |
| **10** | **Trees & Ensembles** | [PDF] | [`10_trees_ensembles.ipynb`](https://www.google.com/search?q=notebooks/10_trees_ensembles.ipynb) |
| **11** | **XAI & Experimentation** | [PDF] | [`11_xai_experimentation.ipynb`](https://www.google.com/search?q=notebooks/11_xai_experimentation.ipynb) |
| **Bonus** | 🎨 Generative models (VAE) | — | [`12_bonus_generative_models_vae.ipynb`](https://www.google.com/search?q=notebooks/12_bonus_generative_models_vae.ipynb) |

> *Slides are available for download on DTU Learn.*

---

## 📦 Pretrained Artifacts

The `models/` directory contains lightweight pretrained weights. This ensures that computationally heavy sessions (like **Lecture 07: CNN**) can be run interactively without long wait times during class.

To retrain models from scratch, check the `scripts/` directory.

---

## ⚖️ License

Distributed under the **MIT License**. See [`LICENSE`](https://www.google.com/search?q=LICENSE) for more information.

<div align="center">
<sub>Designed for the Applied Machine Learning Course at DTU</sub>
</div>



**Do you want me to help you create a specific `requirements.txt` based on the libraries mentioned in the lecture map (like PyTorch, Pandas, Scikit-Learn)?**

```
