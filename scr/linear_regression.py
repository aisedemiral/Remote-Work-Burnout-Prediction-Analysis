import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

df = pd.read_csv("wfh_burnout_dataset.csv")
df = df.drop(columns=["user_id"])

risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
df["burnout_risk"] = df["burnout_risk"].map(risk_map)
df["day_type"] = LabelEncoder().fit_transform(df["day_type"])

X = df.drop(["burnout_risk", "burnout_score"], axis=1)
y_score = df["burnout_score"]
y_class = df["burnout_risk"]

X_train, X_test, y_score_train, y_score_test, y_class_train, y_class_test = train_test_split(
    X, y_score, y_class, test_size=0.20, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_score_train)
y_pred_score = lr.predict(X_test)

y_pred_class = np.digitize(y_pred_score, bins=[3.5, 6.5]) 

accuracy = accuracy_score(y_class_test, y_pred_class)
precision, recall, f1, _ = precision_recall_fscore_support(y_class_test, y_pred_class, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
