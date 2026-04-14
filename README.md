# Surrogate Modeling of a Binary Distillation Column
## FOSSEE Screening Task - Task 3

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-green.svg)
![Chemical Engineering](https://img.shields.io/badge/Core-Chemical%20Engineering-orange.svg)

### 📊 Project Overview
This repository contains the implementation of a **Surrogate Model** for a binary distillation column (Benzene-Toluene system). The goal is to predict critical column performance variables using data-driven machine learning models, derived from physically consistent simulations using the **Fenske-Underwood-Gilliland (FUG)** shortcut method.

### 🔬 Technical Specifications
- **System:** Binary mixture of Benzene (Light Key) and Toluene (Heavy Key).
- **Thermodynamic Model:** Peng-Robinson (PR) Equation of State.
- **Simulation Method:** FUG Shortcut Equations for calculating minimum stages, minimum reflux, and optimal feed stage.
- **Input Variables:** Feed Temperature, Pressure, Composition, No. of Stages, Feed Stage, Reflux Ratio, Bottoms Rate.
- **Target Variables:** Distillate Purity ($x_D$), Bottoms Purity ($x_B$), Condenser Duty ($Q_C$), Reboiler Duty ($Q_R$).

### 🤖 Machine Learning Models
We implement and compare five different regression architectures to identify the most robust surrogate:
1. **Polynomial Regression:** Baseline model with 2nd-degree features.
2. **Random Forest:** Ensemble of decision trees for non-linear capturing.
3. **XGBoost:** Gradient-boosted trees for high efficiency.
4. **Support Vector Regression (SVR):** RBF kernel-based regression.
5. **Artificial Neural Network (ANN):** Multi-layer Perceptron (MLP) for complex mapping.

### 📁 Directory Structure
```text
├── DWSim/              # Flowsheet and simulation files
├── data/               # Generated dataset (dataset.csv)
├── notebooks/          # Jupyter notebooks for EDA and Modeling
│   └── ml_model.ipynb  # Primary modeling and evaluation notebook
├── scripts/            # Python scripts for data generation
│   └── generate_dataset.py
├── results/            # Performance plots and evaluation metrics
└── report/             # Final Technical Report
```

### 🚀 Getting Started
1. **Install Dependencies:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn xgboost
   ```
2. **Generate Data:**
   Run `scripts/generate_dataset.py` to create a new simulation dataset based on FUG correlations.
3. **Train Models:**
   Open `notebooks/ml_model.ipynb` to execute the EDA and model training pipeline.

### 📈 Results
The surrogate models achieve high accuracy on the test set, with **Random Forest** and **ANN** typically showing the best performance for predicting heat duties and component compositions respectively.

---
**Author:** Yusuf Mustafa Ali Malu Bhai Wala
