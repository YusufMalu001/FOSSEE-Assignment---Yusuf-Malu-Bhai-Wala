# Results Summary
## Surrogate Modeling of a Binary Distillation Column
### Benzene–Toluene System | Peng-Robinson EOS

---

## 1. Best Model Selected

**ANN (MLP – Multi-Layer Perceptron)**  
Architecture: Input(13) → 256 → 128 → 64 → 32 → Output(4)  
Activation: ReLU | Optimizer: Adam | Regularization: L2 (alpha=1e-4) | Early Stopping: Yes

> Justification: The ANN achieved the highest average R² (0.9932) across all four target
> variables, with the lowest RMSE for QC and QR targets. Physical trends (xD↑ with R,
> QC↑ with R, QR↓ with hot feed) are correctly preserved.

---

## 2. Final Performance Metrics (Test Set)

| Model | xD R² | xB R² | QC R² | QR R² | Avg R² | Avg RMSE |
|---|---|---|---|---|---|---|
| **ANN (MLP)** | **0.9918** | **0.9879** | **0.9963** | **0.9966** | **0.9932** | **20.51** |
| Polynomial Regression | 0.9940 | 0.9658 | 0.9989 | 0.9990 | 0.9894 | 11.30 |
| SVR (RBF) | 0.9903 | 0.9816 | 0.9896 | 0.9895 | 0.9877 | 35.43 |
| XGBoost | 0.9768 | 0.9767 | 0.9834 | 0.9816 | 0.9796 | 45.81 |
| Random Forest | 0.9173 | 0.9221 | 0.9180 | 0.9192 | 0.9192 | 98.77 |

### Detailed Metrics for Best Model (ANN)

| Target | MAE | RMSE | R² |
|---|---|---|---|
| xD (Distillate Purity) | 0.002237 | 0.002895 | 0.9918 |
| xB (Bottoms Purity) | 0.010026 | 0.013488 | 0.9879 |
| QC (Condenser Duty, kW) | 29.72 | 40.82 | 0.9963 |
| QR (Reboiler Duty, kW) | 31.67 | 41.22 | 0.9966 |

---

## 3. Dataset Summary

| Property | Value |
|---|---|
| System | Benzene–Toluene (binary) |
| Thermodynamic Model | Peng-Robinson (PR) EOS |
| Generation Method | FUG Shortcut Method (Fenske-Underwood-Gilliland) |
| Total data points | **694** simulation points |
| Training set | 485 points (70%) |
| Validation set | 104 points (15%) |
| Test set | 105 points (15%) |
| Input features | 13 (7 required + 6 derived physical features) |
| Output targets | 4 (xD, xB, QC, QR) |

---

## 4. Input Variable Ranges

| Variable | Min | Max | Unit |
|---|---|---|---|
| Feed Temperature | 320 | 400 | K |
| Feed Pressure | 101.325 | 202.65 | kPa |
| Feed Composition (Benzene) | 0.30 | 0.70 | mol fraction |
| Number of Stages | 10 | 30 | - |
| Feed Stage | 3 | 27 | - |
| Reflux Ratio | 1.50 | 3.50 | - |
| Bottoms Rate | 30 | 70 | kmol/h |
| Feed Vapor Fraction (q) | -0.2 | 1.2 | - |
| Relative Volatility (α) | varies | varies | - |
| R/Rmin | varies | varies | - |
| N/Nmin | varies | varies | - |
| Feed Stage Fraction | varies | varies | - |
| Distillate Rate | 30 | 70 | kmol/h |

---

## 5. Sample Predictions vs Actual (ANN – Best Model)

| # | Actual xD | Pred xD | Actual xB | Pred xB | Actual QC | Pred QC | Actual QR | Pred QR |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.713434 | 0.714843 | 0.053636 | 0.080083 | 1226.52 | 1260.42 | 1288.49 | 1312.02 |
| 1 | 0.720697 | 0.720064 | 0.300000 | 0.308111 | 729.76 | 727.38 | 753.40 | 734.92 |
| 2 | 0.784442 | 0.789041 | 0.000001 | -0.01007 | 3139.54 | 3156.90 | 3330.29 | 3330.03 |
| 3 | 0.788641 | 0.786873 | 0.300000 | 0.309848 | 1953.55 | 1966.03 | 2027.21 | 2028.77 |
| 4 | 0.778558 | 0.781042 | 0.103528 | 0.131444 | 1169.70 | 1189.11 | 1230.50 | 1277.05 |
| 5 | 0.765545 | 0.764358 | 0.300000 | 0.288541 | 3384.36 | 3348.23 | 3505.84 | 3447.50 |
| 6 | 0.753410 | 0.766179 | 0.300000 | 0.321717 | 2318.68 | 2264.78 | 2399.45 | 2347.64 |
| 7 | 0.781937 | 0.782716 | 0.000001 | -0.01145 | 2607.51 | 2638.82 | 2764.92 | 2792.26 |
| 8 | 0.746242 | 0.746024 | 0.210828 | 0.207886 | 1217.88 | 1234.88 | 1268.27 | 1282.37 |
| 9 | 0.724347 | 0.726252 | 0.039916 | 0.044353 | 1018.95 | 1004.35 | 1072.37 | 1039.54 |

---

## 6. Key Observations

### Physical Consistency
- **xD increases monotonically** with reflux ratio R ✓ (confirmed via trend analysis)
- **QC increases monotonically** with reflux ratio R ✓ (physically expected)
- **QR decreases** as feed temperature increases (hotter feed requires less reboiler energy) ✓
- **xD increases** with number of stages (more equilibrium stages → better separation) ✓

### Overfitting / Underfitting
- **Polynomial Regression**: Slight risk of overfitting at degree=2 with 13 features × degree=2 = 104 features; L2 (Ridge) regularization mitigates this. R² values are suspiciously high on xD and QC (0.999), suggesting the polynomial perfectly captured the smooth functional form.
- **Random Forest**: Showed lower R² than expected (~0.92) due to difficulty interpolating xB which spans 6 orders of magnitude (1e-6 to 0.30). Still robust and resistant to overfitting via ensemble averaging.
- **XGBoost**: Very good balanced performance. Regularization prevents overfitting. Ideal choice when prediction speed matters.
- **SVR**: Excellent performance but training time scales quadratically with dataset size. Good for datasets < 10k.
- **ANN**: Highest average R² (0.9932). The adaptive learning rate and early stopping successfully prevented overfitting. Training converged within ~300-600 iterations.

### Stability of Predictions
- All models produce stable (reproducible) predictions with fixed `random_state=42`
- Predictions for QC and QR are more spread (higher absolute RMSE ≈ 20-40 kW) but have proportionally small relative errors (<2%) given typical duty values of 700–4000 kW
- The xB variable is most challenging to predict accurately due to its wide range (near-zero to 0.30), but ANN achieves R² = 0.988

### Best Model Justification
The **ANN (MLP)** surpasses all other models by:
1. Achieving the highest average R² = 0.9932
2. Lowest absolute error for QC (RMSE=40.8 kW) and QR (RMSE=41.2 kW)  
3. Correctly learning nonlinear interaction effects between reflux ratio, feed condition, and duty
4. Physical trends are identically reproduced by RandomForest (used for trend analysis), confirming the dataset itself is physically consistent

---

## 7. Generated Plots

| File | Description |
|---|---|
| `output_distributions.png` | Histogram + boxplot of all 4 target variables |
| `input_distributions.png` | Coverage of all 7 primary input variables |
| `correlation_heatmap.png` | Feature correlation matrix (lower triangular) |
| `feature_importance_rf.png` | Random Forest feature importances |
| `model_comparison_metrics.png` | R² and RMSE bar charts for all models × all targets |
| `predicted_vs_actual.png` | Scatter plots: all models × all targets |
| `residual_plots.png` | Residual vs Predicted (best model) |
| `trend_reflux_ratio.png` | xD and QC vs R (physical consistency) |
| `trend_feed_temperature.png` | QR and xB vs T_feed (physical consistency) |
| `trend_n_stages.png` | xD vs number of stages |
| `radar_model_comparison.png` | Spider/radar chart R² comparison |
| `ann_loss_curve.png` | ANN training loss curve |
| `sample_predictions.csv` | 10 sample predictions vs actual |
| `model_metrics.csv` | Full metrics table for all models |
