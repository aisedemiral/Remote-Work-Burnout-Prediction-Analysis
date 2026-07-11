import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('wfh_burnout_dataset.csv')

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.scatterplot(data=df, x='sleep_hours', y='fatigue_score', 
                color='#6495ED', alpha=0.6, s=60)

plt.title('Uyku Süresi ve Yorgunluk Skoru Arasındaki İlişki', fontsize=14)
plt.xlabel('Uyku Süresi (Saat)', fontsize=12)
plt.ylabel('Yorgunluk Skoru', fontsize=12)

plt.tight_layout()
plt.show()
