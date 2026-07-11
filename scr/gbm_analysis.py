import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbm.fit(X_train_smote, y_train_smote)
y_pred = gbm.predict(X_test)

plt.figure(figsize=(10, 6))
feat_importances = pd.Series(gbm.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh', color='teal')
plt.title("Öznitelik Önemi (Feature Importance)")
plt.xlabel("Skor")
plt.savefig("gbm_feature_importance.pdf") 
plt.show()

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title("Hata Matrisi (Confusion Matrix)")
plt.savefig("gbm_confusion_matrix.pdf")
plt.show()

print("--- Gradient Boosting Performans ---")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
