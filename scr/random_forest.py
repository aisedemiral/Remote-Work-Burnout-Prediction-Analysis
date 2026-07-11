import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

df = pd.read_csv("wfh_burnout_dataset.csv")
df = df.drop(columns=["user_id", "burnout_score"])

risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
df["burnout_risk"] = df["burnout_risk"].map(risk_map)
df["day_type"] = LabelEncoder().fit_transform(df["day_type"])

X = df.drop("burnout_risk", axis=1)
y = df["burnout_risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_smote, y_train_smote)
y_pred = rf_model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

feature_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_imp)

plt.figure(figsize=(10,6))
sns.barplot(x=feature_imp, y=feature_imp.index, hue=feature_imp.index, palette='magma', legend=False)
plt.title("Random Forest - Ozellik Onem Duzeyi (Feature Importance)")
plt.savefig("rf_feature_importance.pdf")

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek Değer")
plt.savefig("rf_confusion_matrix.pdf")
plt.show()
