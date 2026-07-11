import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

df = pd.read_csv("wfh_burnout_dataset.csv")

df['is_over_10_hours'] = df['work_hours'].apply(lambda x: 'Over 10h' if x > 10 else 'Under 10h')

contingency_table = pd.crosstab(df['is_over_10_hours'], df['burnout_risk'])
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square Testi P-Değeri: {p:.5f}")
if p < 0.05:
    print("Sonuç: Çalışma süresi ile tükenmişlik riski arasında istatistiksel olarak anlamlı bir ilişki vardır.")


plt.figure(figsize=(8, 6))
sns.countplot(x='is_over_10_hours', hue='burnout_risk', data=df, palette='viridis')
plt.title('Çalışma Süresi (10 Saat Eşiği) ve Tükenmişlik Riski İlişkisi', fontsize=14)
plt.ylabel('Çalışan Sayısı')
plt.xlabel('Günlük Çalışma Süresi')
plt.savefig("work_hours_threshold.pdf")
plt.show()
