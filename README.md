# ADNI MRI Analysis and ANN Classification

## 📌 Project Overview

This project analyzes MRI-based features from the ADNI dataset and explores their relationship with diagnostic categories.

The project combines statistical analysis, feature selection, dimensionality reduction, and Artificial Neural Network (ANN) classification.

## 🎯 Objectives

- Analyze selected MRI features using descriptive statistics.
- Assess feature distributions using the Shapiro-Wilk normality test.
- Analyze skewness and kurtosis.
- Compare MRI features across diagnostic groups using ANOVA.
- Compare simple random sampling and stratified sampling.
- Analyze hippocampus volume across follow-up visits.
- Identify important features using Decision Tree feature importance.
- Perform Chi-Square feature selection.
- Reduce feature dimensionality using Principal Component Analysis (PCA).
- Build an Artificial Neural Network (ANN) for classification.
- Evaluate the ANN using accuracy, classification report, and confusion matrix.

## 🧠 MRI Features

The analysis uses the following baseline MRI features:

- Ventricles
- Hippocampus
- Whole Brain
- Entorhinal
- Fusiform
- Mid Temporal
- Intracranial Volume (ICV)

## 🔬 Methodology

The project follows these main steps:

1. Data loading and preparation
2. Missing-value handling
3. Descriptive statistical analysis
4. Normality testing
5. Skewness and kurtosis analysis
6. ANOVA analysis
7. Sampling comparison
8. Feature importance analysis
9. Chi-Square feature selection
10. PCA dimensionality reduction
11. ANN classification
12. Model evaluation

## 🤖 Machine Learning

The ANN model uses:

- Multi-Layer Perceptron (MLP)
- ReLU activation
- Adam optimizer
- Two hidden layers: 64 and 32 neurons
- PCA-reduced features

## 📊 Evaluation

The classification model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## 🛠 Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Google Colab

## 📁 Project Structure

```text
adni-mri-disease-classification/
│
├── README.md
├── main.py
└── requirements.txt
