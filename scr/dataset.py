
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("wfh_burnout_dataset.csv")

print("İlk 5 Satır")
print(df.head())

print("\nVeri Seti Boyutu")
print(df.shape)

print("\nVeri Tipleri")
df.info()

print("\nEksik Veri Sayıları")
print(df.isnull().sum())

print("\nİstatistiksel Bilgiler")
print(df.describe())

print("\nBurnout Risk Dağılımı")
print(df["burnout_risk"].value_counts())

print("\nYüzdelik Dağılım")
print(df["burnout_risk"].value_counts(normalize=True) * 100)
print(df.groupby("burnout_risk")["burnout_score"].describe())
# Grafik
df["burnout_risk"].value_counts().plot(kind="bar")
plt.title("Burnout Risk Distribution")
plt.xlabel("Burnout Risk")
plt.ylabel("Count")
plt.show()
