

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro, f_oneway

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from google.colab import files



print("Uploading dataset...")

uploaded = files.upload()

file_name = next(iter(uploaded))

df = pd.read_excel(file_name)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)



features = [
    "Ventricles_bl",
    "Hippocampus_bl",
    "WholeBrain_bl",
    "Entorhinal_bl",
    "Fusiform_bl",
    "MidTemp_bl",
    "ICV_bl"
]

target = "DX"



X = df[features].copy()
y = df[target].copy()

# Convert features to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Fill missing values using column means
X = X.fillna(X.mean())

print("\nSelected features:")
print(features)

print("\nTarget distribution:")
print(y.value_counts())



print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(X.describe())


# ============================================================
# 5. Normality Assessment - Shapiro-Wilk Test
# ============================================================

print("\n" + "=" * 60)
print("NORMALITY TEST - SHAPIRO-WILK")
print("=" * 60)

normality_results = []

for col in features:

    # Histogram
    plt.figure(figsize=(7, 4))

    sns.histplot(
        X[col],
        kde=True
    )

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    # Shapiro-Wilk test
    stat, p = shapiro(X[col])

    result = "Normal Distribution" if p > 0.05 else "Not Normal Distribution"

    normality_results.append([
        col,
        stat,
        p,
        result
    ])

    print(f"\nFeature: {col}")
    print(f"Statistic = {stat:.4f}")
    print(f"P-value = {p:.6f}")
    print(f"Result = {result}")


normality_df = pd.DataFrame(
    normality_results,
    columns=[
        "Feature",
        "Statistic",
        "P-value",
        "Result"
    ]
)

print("\nNormality Summary:")
print(normality_df)


# ============================================================
# 6. Skewness and Kurtosis
# ============================================================

print("\n" + "=" * 60)
print("SKEWNESS")
print("=" * 60)

print(X.skew())


print("\n" + "=" * 60)
print("KURTOSIS")
print("=" * 60)

print(X.kurtosis())


# ============================================================
# 7. Boxplot Analysis
# ============================================================

print("\n" + "=" * 60)
print("STANDARDIZED BOXPLOTS")
print("=" * 60)

scaler_box = StandardScaler()

X_box = pd.DataFrame(
    scaler_box.fit_transform(X),
    columns=features
)

plt.figure(figsize=(12, 6))

sns.boxplot(data=X_box)

plt.title("Standardized Boxplots of MRI Features")
plt.xlabel("MRI Features")
plt.ylabel("Standardized Values")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ============================================================
# 8. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ============================================================
# 9. Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 10. ANOVA Test
# ============================================================

print("\n" + "=" * 60)
print("ANOVA TEST")
print("=" * 60)

anova_results = []

for col in features:

    groups = [
        X.loc[y == category, col]
        for category in y.unique()
    ]

    stat, p = f_oneway(*groups)

    result = "Significant Effect on DX" if p < 0.05 else "Not Significant"

    anova_results.append([
        col,
        stat,
        p,
        result
    ])

    print(f"\nFeature: {col}")
    print(f"Statistic = {stat:.4f}")
    print(f"P-value = {p:.6f}")
    print(f"Result = {result}")


anova_df = pd.DataFrame(
    anova_results,
    columns=[
        "Feature",
        "F-Statistic",
        "P-value",
        "Result"
    ]
)

print("\nANOVA Summary:")
print(anova_df)


# ============================================================
# 11. Sampling Methods
# ============================================================

print("\n" + "=" * 60)
print("SAMPLING METHODS")
print("=" * 60)


# Simple Random Sampling
simple_sample = df.sample(
    frac=0.70,
    random_state=42
)


# Stratified Sampling
strat_sample, _ = train_test_split(
    df,
    test_size=0.30,
    stratify=df[target],
    random_state=42
)


print("\nOriginal Distribution:")
print(df[target].value_counts(normalize=True))


print("\nSimple Random Sample:")
print(simple_sample[target].value_counts(normalize=True))


print("\nStratified Sample:")
print(strat_sample[target].value_counts(normalize=True))


# ============================================================
# 12. Mean Comparison
# ============================================================

compare_features = [
    "Hippocampus_bl",
    "Ventricles_bl",
    "WholeBrain_bl"
]

print("\n" + "=" * 60)
print("MEAN COMPARISON")
print("=" * 60)

print("\nOriginal Mean:")
print(df[compare_features].mean())

print("\nSimple Sample Mean:")
print(simple_sample[compare_features].mean())

print("\nStratified Sample Mean:")
print(strat_sample[compare_features].mean())


# ============================================================
# 13. Standard Deviation Comparison
# ============================================================

print("\n" + "=" * 60)
print("STANDARD DEVIATION COMPARISON")
print("=" * 60)

print("\nOriginal Standard Deviation:")
print(df[compare_features].std())

print("\nSimple Sample Standard Deviation:")
print(simple_sample[compare_features].std())

print("\nStratified Sample Standard Deviation:")
print(strat_sample[compare_features].std())


# ============================================================
# 14. Visit Progression Analysis
# ============================================================

print("\n" + "=" * 60)
print("VISIT PROGRESSION ANALYSIS")
print("=" * 60)


if "VISCODE" in df.columns:

    visit_order = {
        "bl": 0,
        "m03": 3,
        "m06": 6,
        "m12": 12,
        "m18": 18,
        "m24": 24,
        "m36": 36,
        "m48": 48,
        "m60": 60,
        "m72": 72,
        "m84": 84,
        "m96": 96
    }

    ts = (
        df.groupby("VISCODE")["Hippocampus_bl"]
        .mean()
        .reset_index()
    )

    ts["Months"] = ts["VISCODE"].map(visit_order)

    ts = ts.dropna(
        subset=["Months"]
    )

    ts = ts.sort_values(
        "Months"
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        ts["Months"],
        ts["Hippocampus_bl"],
        marker="o",
        linewidth=2
    )

    for i in range(len(ts)):

        plt.annotate(
            ts["VISCODE"].iloc[i],
            (
                ts["Months"].iloc[i],
                ts["Hippocampus_bl"].iloc[i]
            )
        )

    plt.title(
        "Mean Hippocampus Volume Across Follow-up Visits"
    )

    plt.xlabel(
        "Time Since Baseline (Months)"
    )

    plt.ylabel(
        "Mean Hippocampus Volume"
    )

    plt.xticks(
        ts["Months"]
    )

    plt.grid(True)

    plt.tight_layout()
    plt.show()

else:

    print("VISCODE column not found.")


# ============================================================
# 15. Null Hypothesis
# ============================================================

print("\n" + "=" * 60)
print("NULL HYPOTHESIS")
print("=" * 60)

print(
    "H0: Changes in Hippocampus volume over time "
    "are random and do not show a systematic trend."
)


# ============================================================
# PART 2
# Feature Selection and ANN Classification
# ============================================================


# ============================================================
# 16. Encode Target Variable
# ============================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)


print("\n" + "=" * 60)
print("TARGET ENCODING")
print("=" * 60)

print("Classes:")
print(encoder.classes_)


# ============================================================
# 17. Decision Tree Feature Importance
# ============================================================

print("\n" + "=" * 60)
print("DECISION TREE FEATURE IMPORTANCE")
print("=" * 60)

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(
    X,
    y_encoded
)

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": dt.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df)


plt.figure(figsize=(8, 5))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title(
    "Decision Tree Feature Importance"
)

plt.tight_layout()
plt.show()


# ============================================================
# 18. Chi-Square Feature Selection
# ============================================================

print("\n" + "=" * 60)
print("CHI-SQUARE FEATURE SELECTION")
print("=" * 60)

scaler_chi = MinMaxScaler()

X_chi = scaler_chi.fit_transform(X)

chi_scores, p_values = chi2(
    X_chi,
    y_encoded
)

chi_df = pd.DataFrame({
    "Feature": features,
    "Chi2 Score": chi_scores,
    "P Value": p_values
})

chi_df = chi_df.sort_values(
    by="Chi2 Score",
    ascending=False
)

print(chi_df)


plt.figure(figsize=(8, 5))

sns.barplot(
    data=chi_df,
    x="Chi2 Score",
    y="Feature"
)

plt.title(
    "Chi-Square Feature Scores"
)

plt.tight_layout()
plt.show()


# ============================================================
# 19. Principal Component Analysis - PCA
# ============================================================

print("\n" + "=" * 60)
print("PCA ANALYSIS")
print("=" * 60)

scaler_pca = StandardScaler()

X_scaled = scaler_pca.fit_transform(X)

pca = PCA()

X_pca = pca.fit_transform(
    X_scaled
)

explained_var = (
    pca.explained_variance_ratio_
)

cumulative_var = np.cumsum(
    explained_var
)


# Scree Plot

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(explained_var) + 1),
    explained_var,
    marker="o"
)

plt.title(
    "Scree Plot"
)

plt.xlabel(
    "Principal Component"
)

plt.ylabel(
    "Explained Variance Ratio"
)

plt.grid(True)

plt.tight_layout()
plt.show()


# Cumulative Variance Plot

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(cumulative_var) + 1),
    cumulative_var,
    marker="o"
)

plt.axhline(
    y=0.75,
    linestyle="--"
)

plt.title(
    "Cumulative Explained Variance"
)

plt.xlabel(
    "Number of Components"
)

plt.ylabel(
    "Cumulative Variance"
)

plt.grid(True)

plt.tight_layout()
plt.show()


# Number of components explaining at least 75%

n_components = (
    np.argmax(
        cumulative_var >= 0.75
    ) + 1
)

print(
    "\nNumber of Components Retained:",
    n_components
)


# PCA Results Table

pca_table = pd.DataFrame({
    "Component": [
        f"PC{i + 1}"
        for i in range(len(explained_var))
    ],
    "Explained Variance": explained_var,
    "Cumulative Variance": cumulative_var
})

print("\nPCA Results:")
print(pca_table)


# Reduced Dataset

pca_final = PCA(
    n_components=n_components
)

X_reduced = pca_final.fit_transform(
    X_scaled
)


# ============================================================
# 20. ANN Classification
# ============================================================

print("\n" + "=" * 60)
print("ANN CLASSIFICATION")
print("=" * 60)


X_train_ann, X_test_ann, y_train_ann, y_test_ann = (
    train_test_split(
        X_reduced,
        y_encoded,
        test_size=0.30,
        random_state=42,
        stratify=y_encoded
    )
)


ann = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)


ann.fit(
    X_train_ann,
    y_train_ann
)


y_pred = ann.predict(
    X_test_ann
)


# ============================================================
# 21. ANN Accuracy
# ============================================================

accuracy = accuracy_score(
    y_test_ann,
    y_pred
)

print(
    f"\nANN Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 22. Classification Report
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test_ann,
        y_pred,
        target_names=encoder.classes_
    )
)


# ============================================================
# 23. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test_ann,
    y_pred
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "ANN Confusion Matrix"
)

plt.tight_layout()
plt.show()


# ============================================================
# END OF PROJECT
# ============================================================

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)
