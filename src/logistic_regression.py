import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

df = pd.read_csv("wfh_burnout_dataset.csv")
df = df.drop(columns=["user_id", "burnout_score"])

risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
df["burnout_risk"] = df["burnout_risk"].map(risk_map)
df["day_type"] = LabelEncoder().fit_transform(df["day_type"])

X = df.drop("burnout_risk", axis=1)
y = df["burnout_risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_smote, y_train_smote)
y_pred = model.predict(X_test_scaled)

print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.savefig("lr_confusion_matrix.pdf") 

coef_df = pd.DataFrame({'Degisken': X.columns, 'Katsayi': model.coef_[0]})
coef_df = coef_df.sort_values(by='Katsayi', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Katsayi', y='Degisken', data=coef_df, hue='Degisken', palette='viridis', legend=False)
plt.title("Lojistik Regresyon - Ozellik Etki Katsayilari")
plt.savefig("lr_feature_importance.pdf")
plt.show()
