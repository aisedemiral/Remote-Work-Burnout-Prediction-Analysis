import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE

np.random.seed(42)

df = pd.read_csv("wfh_burnout_dataset.csv")

df = df.drop(columns=["user_id", "burnout_score"])

risk_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

df["burnout_risk"] = df["burnout_risk"].map(risk_map)

le = LabelEncoder()
df["day_type"] = le.fit_transform(df["day_type"])

X = df.drop("burnout_risk", axis=1)
y = df["burnout_risk"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


results = []

for name, model in models.items():

    model.fit(X_train_smote, y_train_smote)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro"
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1
    })

results_df = pd.DataFrame(results)

print("\n=== MODEL COMPARISON TABLE ===\n")
print(results_df.round(4).to_string(index=False))

plot_df = results_df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(11, 6))

ax = sns.barplot(
    data=plot_df,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title(
    "Performance Comparison of Machine Learning Models",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel("Score")
plt.xlabel("")

plt.ylim(0.80, 1.00)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.3f",
        padding=2
    )

plt.tight_layout()

plt.savefig(
    "model_comparison_chart.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
