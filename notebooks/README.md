# Notebooks

Each notebook corresponds to a lecture (plus a few prerequisites and a bonus session).

## Recommended order

1. `00_prerequisites/00_python_numpy_crash_course.ipynb` (optional)
2. `00_prerequisites/01_pandas_intro.ipynb` (optional)
3. `01_introduction.ipynb`
4. `02_ml_foundations.ipynb`
5. `03_regression.ipynb`
6. `04_classification.ipynb`
7. `05_clustering.ipynb`
8. `06_neural_networks.ipynb`
9. `07_cnn.ipynb`
10. `08_resampling.ipynb`
11. `09_svm.ipynb`
12. `10_trees_ensembles.ipynb`
13. `11_xai_experimentation.ipynb`
14. `12_bonus_generative_models_vae.ipynb` (bonus)

## A note on paths

Every notebook starts with a **Setup** cell that:
- finds the repository root,
- `chdir`s to it (so relative paths work consistently), and
- defines `DATA_DIR`, `FIGURES_DIR`, and `MODELS_DIR`.

When a notebook saves plots, it writes into `./pictures/` by default.
