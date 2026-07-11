import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv("wfh_burnout_dataset.csv")
df = df.drop(columns=["user_id", "burnout_score"])
risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
df["burnout_risk"] = df["burnout_risk"].map(risk_map)
df["day_type"] = LabelEncoder().fit_transform(df["day_type"])

X = df.drop("burnout_risk", axis=1)
y = df["burnout_risk"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_smote, y_train_smote)

y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

y_pred = svm_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.title("SVM - Confusion Matrix")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek Değer")
plt.savefig("svm_confusion_matrix.pdf")
plt.show()

result = permutation_importance(svm_model, X_test, y_test, n_repeats=10, random_state=42)
perm_imp = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=perm_imp, y=perm_imp.index, hue=perm_imp.index, palette='viridis', legend=False)
plt.title("SVM - Ozellik Onem Duzeyi (Permutation Importance)")
plt.savefig("svm_feature_importance.pdf")
plt.show()
