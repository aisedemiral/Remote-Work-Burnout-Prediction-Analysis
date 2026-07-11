import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("wfh_burnout_dataset.csv")

df = df.drop(columns=[
    "user_id",
    "burnout_score"
])

le_day = LabelEncoder()
df["day_type"] = le_day.fit_transform(df["day_type"])

risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
df["burnout_risk"] = df["burnout_risk"].map(risk_map)

X = df.drop("burnout_risk", axis=1)
y = df["burnout_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("\nTrain Sınıf Dağılımı")
print(y_train.value_counts())


from imblearn.over_sampling import SMOTE


smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nSMOTE Sonrası Eğitim Dağılımı")

print(y_train_smote.value_counts())

print("\nTest Sınıf Dağılımı")
print(y_test.value_counts())
