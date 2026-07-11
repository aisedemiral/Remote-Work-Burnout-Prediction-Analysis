import pandas as pd  
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("wfh_burnout_dataset.csv") 

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

features = ['fatigue_score', 'isolation_index', 'sleep_hours', 'work_hours']
titles = ['Yorgunluk Skoru', 'İzolasyon İndeksi', 'Uyku Süresi', 'Çalışma Saatleri']

for i, feature in enumerate(features):
    sns.boxplot(x='burnout_risk', y=feature, data=df, palette='viridis', ax=axes[i])
    axes[i].set_title(titles[i], fontsize=14)
    axes[i].set_xlabel('Tükenmişlik Riski', fontsize=12)
    axes[i].set_ylabel(feature.replace('_', ' ').title(), fontsize=12)

plt.tight_layout()
plt.savefig("burnout_risk_drivers.pdf")
plt.show()
