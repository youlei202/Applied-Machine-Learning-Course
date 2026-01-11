# Applied Machine Learning — Course Repository

This repository contains the **slides**, **Jupyter notebooks**, and a small amount of **supporting code** for an Applied Machine Learning course.

- Slides live under DTU Learn.
- Notebooks live under [`notebooks/`](notebooks/).
- Small helper utilities live under [`src/aml_course/`](src/aml_course/).

The material is organized so that each lecture has a corresponding notebook (with a few optional prerequisite notebooks and one bonus session).

## Repository structure

```
.
├── notebooks/       # Lecture notebooks (Jupyter)
├── src/aml_course/  # Helper utilities imported by notebooks
├── scripts/         # Standalone scripts (e.g., training)
├── models/          # Small pretrained artifacts
├── data/            # Datasets (downloaded or placed manually)
├── pictures/        # Figures saved by notebooks
└── docs/            # Extra course material / notes
```

## Getting started

### 1) Create an environment

You can use **pip** or **conda**.

**Option A — pip**

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

**Option B — conda**

```bash
conda env create -f environment.yml
conda activate aml-course
```

Optional dependencies (used in some advanced sessions):

```bash
pip install -r requirements-optional.txt
```

> Notes
> - Installing **PyTorch** and/or **TensorFlow** can be platform-specific. If you run into issues, follow the official installation instructions for your OS/GPU.

### 2) Launch Jupyter

```bash
jupyter lab
```

Open notebooks from the `notebooks/` folder.

Each notebook starts with a **Setup** cell that:
- finds the repository root,
- changes the working directory to the root (so relative paths work consistently), and
- defines common paths: `DATA_DIR`, `FIGURES_DIR`, `MODELS_DIR`.

## Lecture map

| Lecture | Topic | Slides | Notebook |
|---:|---|---|---|
| Prerequisite A | Python + NumPy Crash Course | — | [`00_python_numpy_crash_course.ipynb`](notebooks/00_prerequisites/00_python_numpy_crash_course.ipynb) |
| Prerequisite B | Pandas Essentials | — | [`01_pandas_intro.ipynb`](notebooks/00_prerequisites/01_pandas_intro.ipynb) |
| 01 | Introduction | [`01_Introduction.pdf`] | [`01_introduction.ipynb`](notebooks/01_introduction.ipynb) |
| 02 | ML Foundations | [`02_ML_Foundations.pdf`] | [`02_ml_foundations.ipynb`](notebooks/02_ml_foundations.ipynb) |
| 03 | Regression | [`03_Regression.pdf`] | [`03_regression.ipynb`](notebooks/03_regression.ipynb) |
| 04 | Classification | [`04_Classification.pdf`] | [`04_classification.ipynb`](notebooks/04_classification.ipynb) |
| 05 | Clustering | [`05_Clustering.pdf`] | [`05_clustering.ipynb`](notebooks/05_clustering.ipynb) |
| 06 | Neural Networks | [`06_Neural_Networks.pdf`] | [`06_neural_networks.ipynb`](notebooks/06_neural_networks.ipynb) |
| 07 | CNNs (PyTorch) | [`07_CNN.pdf`] | [`07_cnn.ipynb`](notebooks/07_cnn.ipynb) |
| 08 | Resampling | [`08_Resampling.pdf`] | [`08_resampling.ipynb`](notebooks/08_resampling.ipynb) |
| 09 | SVM | [`09_SVM.pdf`] | [`09_svm.ipynb`](notebooks/09_svm.ipynb) |
| 10 | Trees & Ensembles | [`10_Trees_and_Ensembles.pdf`] | [`10_trees_ensembles.ipynb`](notebooks/10_trees_ensembles.ipynb) |
| 11 | XAI & Experimentation | [`11_XAI_and_Experimentation.pdf`] | [`11_xai_experimentation.ipynb`](notebooks/11_xai_experimentation.ipynb) |
| Bonus | Generative models (VAE) | — | [`12_bonus_generative_models_vae.ipynb`](notebooks/12_bonus_generative_models_vae.ipynb) |

## Pretrained artifacts

`models/` contains small pretrained weights used to keep some sessions fast to run (e.g., Lecture 7 CNN). You can retrain models using scripts under `scripts/`.

## License

MIT — see [`LICENSE`](LICENSE).
