import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

from imblearn.over_sampling import SMOTE

df = pd.read_csv("wfh_burnout_dataset.csv")
df = df.drop(columns=["user_id", "burnout_score"])

risk_map = {"Low": 0, "Medium": 1, "High": 2}
df["burnout_risk"] = df["burnout_risk"].map(risk_map)

encoder = LabelEncoder()
df["day_type"] = encoder.fit_transform(df["day_type"])

X = df.drop("burnout_risk", axis=1)
y = df["burnout_risk"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel="rbf", probability=True),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

fig_cm, axes_cm = plt.subplots(2, 2, figsize=(14, 12))
fig_fi, axes_fi = plt.subplots(2, 2, figsize=(18, 14))

axes_cm = axes_cm.flatten()
axes_fi = axes_fi.flatten()

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes_cm[i])
    axes_cm[i].set_title(name, fontsize=14, fontweight="bold")
    
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=X.columns)
    else:
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        importance = pd.Series(result.importances_mean, index=X.columns)

    importance = importance.sort_values(ascending=False)

    sns.barplot(
        x=importance.values,
        y=importance.index,
        hue=importance.index,  
        palette="viridis",
        legend=False,         
        ax=axes_fi[i]
    )

    axes_fi[i].set_title(name, fontsize=14, fontweight="bold")
    axes_fi[i].set_xlabel("Önem Skoru", fontsize=11, fontweight="bold")

fig_cm.tight_layout()
fig_fi.tight_layout()
plt.show()
