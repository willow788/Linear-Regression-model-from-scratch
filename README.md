<div align="center">

# 🎯 Linear Regression from Scratch

### *Building Machine Learning Foundations, One Gradient at a Time*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-013243.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Validation-F7931E.svg)](https://scikit-learn.org/)

*A comprehensive implementation of Linear Regression with multiple gradient descent methods, polynomial features, and L1 regularization*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Project Structure](#-project-structure) • [Experiment Logs](#-experiment-logs)

</div>

---

## 📖 About The Project

This repository contains a **from-scratch implementation** of Linear Regression, built to deeply understand the mathematics and mechanics behind one of the most fundamental machine learning algorithms.  

### 🎓 What Makes This Special? 

- ✅ **Pure NumPy Implementation** - No black-box ML libraries for core algorithm
- ✅ **Three Gradient Descent Methods** - Batch, Stochastic, and Mini-Batch
- ✅ **Polynomial Feature Engineering** - Up to 2nd degree with interaction terms
- ✅ **L1 Regularization (Lasso)** - Prevent overfitting and feature selection
- ✅ **Early Stopping** - Intelligent training termination
- ✅ **K-Fold Cross-Validation** - Robust model evaluation
- ✅ **Comprehensive Visualizations** - Loss curves, residuals, correlations
- ✅ **Detailed Experiment Logs** - Journey from negative R² to 98%+ accuracy

---

## 🚀 Features

### 🧮 Multiple Gradient Descent Methods

<table>
<tr>
<td width="33%" align="center">
<h4>Batch Gradient Descent</h4>
<p>Uses entire dataset per iteration</p>
<p>✅ Stable convergence</p>
<p>✅ Smooth loss curves</p>
</td>
<td width="33%" align="center">
<h4>Stochastic Gradient Descent</h4>
<p>One sample at a time</p>
<p>✅ Fast updates</p>
<p>✅ Escapes local minima</p>
</td>
<td width="33%" align="center">
<h4>Mini-Batch GD</h4>
<p>Best of both worlds</p>
<p>✅ Balanced speed</p>
<p>✅ Memory efficient</p>
</td>
</tr>
</table>

### 📊 Advanced Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Polynomial Features** | TV², Radio², TV×Radio, etc. | Captures non-linear relationships |
| **L1 Regularization** | Lasso penalty on weights | Prevents overfitting, feature selection |
| **Z-Score Normalization** | Standardizes features and targets | Faster convergence, stable gradients |
| **Early Stopping** | Monitors loss with patience | Prevents unnecessary iterations |
| **K-Fold CV** | 5-fold cross-validation | Robust performance estimation |

---

## 📦 Installation

### Prerequisites

```bash
Python 3.8+
pip package manager
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/willow788/Linear-Regression-model-from-scratch.git

# Navigate to project directory
cd Linear-Regression-model-from-scratch

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Dependencies

```python
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.23.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

---

## 🎯 Usage

### Quick Example

```python
from linear_regression import LinearRegression
from data_preprocessing import load_and_preprocess_data

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data('Advertising.csv')

# Initialize model
model = LinearRegression(
    learn_rate=0.02,
    iter=50000,
    method='batch',
    l1_reg=0.1
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### Running the Complete Pipeline

```bash
# Run the main script
python main.py
```

### Jupyter Notebook Exploration

```bash
# Launch Jupyter
jupyter notebook

# Open any version notebook
# Navigate to Version- 9/Raw jupyter Notebook/sales. ipynb
```

---

## 📁 Project Structure

```
Linear-Regression-model-from-scratch/
│
├── 📂 Version- 1/                    # Initial experiments
│   └── experiment_log.txt            # Detailed notes on failures and learnings
│
├── 📂 Version- 2/                    # Feature engineering experiments
│   └── experiment_log.txt
│
├── 📂 Version- 3/                    # Normalization improvements
│   └── experiment_log.txt
│
├── 📂 Version- 9/                    # Final optimized version
│   ├── Raw jupyter Notebook/
│   │   └── sales.ipynb              # Complete analysis notebook
│   └── Python Files/
│       ├── data_preprocessing.py    # Data loading and feature engineering
│       ├── linear_regression.py     # Core model implementation
│       ├── model_evaluation.py      # Metrics and cross-validation
│       ├── visualization.py         # Plotting utilities
│       ├── main.py                  # Main execution script
│       └── config.py                # Configuration parameters
│
├── 📊 Advertising.csv                # Dataset
└── 📖 README.md
```

---

## 🧪 Experiment Logs

<details>
<summary><b>🔴 Version 1: The Negative R² Crisis</b></summary>

### Problem
- **R² Score: -18.77** 😱
- Model performing worse than predicting mean

### Root Causes Discovered
1. No feature normalization
2. Learning rate too high causing divergence
3. Basic linear features insufficient for non-linear relationships

### Key Learnings
> "Sometimes you need to fail spectacularly to understand the fundamentals."

</details>

<details>
<summary><b>🟡 Version 2-3: Feature Engineering Journey</b></summary>

### Experiments Conducted
- Added polynomial features (TV², Radio², Newspaper²)
- Implemented interaction terms (TV×Radio, etc.)
- Introduced Z-score normalization
- Tuned learning rates systematically

### Results
- R² improved to ~0.85
- Still experiencing some instability

</details>

<details>
<summary><b>🟢 Version 9: Production-Ready Model</b></summary>

### Final Optimizations
✅ **Z-score normalization** for features and target  
✅ **L1 regularization** (λ = 0.1-0.2)  
✅ **Early stopping** with patience = 1000  
✅ **K-fold cross-validation** for robust evaluation  
✅ **Multiple GD methods** for comparison  

### Performance Metrics

| Metric | Batch GD | Stochastic GD | Mini-Batch GD |
|--------|----------|---------------|---------------|
| **Test R²** | 0.9584 | 0.9850 | 0.9874 |
| **Train R²** | 0.9509 | 0.9848 | 0.9860 |
| **RMSE** | 0.2249 | 0.1352 | 0.1238 |
| **MAE** | 0.1533 | 0.1118 | 0.1011 |

### 🎉 Best Model:  Mini-Batch GD
- **R² Score: 98.74%**
- **Batch Size: 16**
- **Learning Rate: 0.01**
- **Iterations: 1000**

</details>

---

## 📊 Visualizations

<div align="center">

### Loss Convergence

The model demonstrates smooth convergence with proper hyperparameters

### Residual Analysis

Residuals show random scatter around zero, indicating good model fit

### Feature Importance

TV advertising shows strongest correlation with sales, followed by Radio

</div>

---

## 🔬 Mathematical Foundation

### Linear Regression Equation

$$\hat{y} = X\mathbf{w} + b$$

### Loss Function (with L1 Regularization)

$$L(\mathbf{w}, b) = \frac{1}{2m}\sum_{i=1}^{m}(h_\mathbf{w}(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2}\sum_{j=1}^{n}|w_j|$$

### Gradient Descent Update Rules

$$\mathbf{w} := \mathbf{w} - \alpha \cdot \frac{1}{m}X^T(X\mathbf{w} - \mathbf{y}) - \alpha \cdot \lambda \cdot \text{sign}(\mathbf{w})$$

$$b := b - \alpha \cdot \frac{1}{m}\sum_{i=1}^{m}(h_\mathbf{w}(x^{(i)}) - y^{(i)})$$

Where: 
- $\alpha$ = learning rate
- $m$ = number of samples
- $\lambda$ = regularization parameter

---

## 📈 Dataset

**Advertising Dataset**
- **Source**: Kaggle/UCI ML Repository
- **Samples**: 200
- **Features**: TV, Radio, Newspaper advertising budgets
- **Target**: Sales figures

### Feature Engineering

Original 3 features expanded to 9:
1. TV
2. Radio
3. Newspaper
4. TV² (squared term)
5. Radio² (squared term)
6. Newspaper² (squared term)
7. TV × Radio (interaction)
8. TV × Newspaper (interaction)
9. Radio × Newspaper (interaction)

---

## 🎓 Key Learnings

### 1. **Data Normalization is Critical**
Without normalization, gradients explode and convergence fails

### 2. **Feature Engineering Matters**
Polynomial and interaction terms capture non-linear relationships

### 3. **Regularization Prevents Overfitting**
L1 penalty keeps weights small and performs feature selection

### 4. **Hyperparameter Tuning is an Art**
Learning rate, regularization, and batch size must be balanced

### 5. **Cross-Validation is Essential**
K-fold CV provides honest performance estimates

---

## 🛠️ Future Improvements

- [ ] Add Elastic Net (L1 + L2)
- [ ] Adaptive learning rates (Adam, RMSprop)
- [ ] Automatic hyperparameter tuning (Grid Search)
- [ ] Feature selection algorithms
- [ ] Support for categorical features
- [ ] Model serialization (save/load)
- [ ] Web interface for predictions

---

## 🤝 Contributing

Contributions are welcome! Feel free to: 

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Advertising dataset from Kaggle
- **Inspiration**: Andrew Ng's Machine Learning course
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn

---

<div align="center">

### ⭐ Star this repo if you found it helpful! 

**Built with 💙 and ☕ by [willow788](https://github.com/willow788)**

*Learning by doing, one line of code at a time*

</div>
