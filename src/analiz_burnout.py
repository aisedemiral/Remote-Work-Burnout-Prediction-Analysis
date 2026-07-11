import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("final_prepared_wfh_burnout.csv")
corr_with_burnout = df.corr(numeric_only=True)[['burnout_score']].sort_values(by='burnout_score', ascending=False)

corr_with_burnout = corr_with_burnout.drop('burnout_score')

plt.figure(figsize=(8, 10))
sns.barplot(x=corr_with_burnout['burnout_score'], y=corr_with_burnout.index, 
            hue=corr_with_burnout.index, palette='coolwarm', legend=False)
plt.title("Burnout Score (Output) ile Girdi Değişkenlerinin Korelasyonu", fontsize=14)
plt.xlabel("Korelasyon Katsayısı (r)", fontsize=12)
plt.ylabel("Girdi Değişkenleri", fontsize=12)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8) 
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("burnout_input_correlation.pdf")
plt.show()
