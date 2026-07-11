import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("final_prepared_wfh_burnout.csv")
cols_to_keep = [
    'work_hours', 'screen_time_hours', 'meetings_count', 'breaks_taken', 
    'after_hours_work', 'app_switches', 'sleep_hours', 'task_completion', 
    'isolation_index', 'fatigue_score'
]
numeric_df = df[cols_to_keep]

corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt=".2f", 
    annot_kws={"size": 10},
    vmin=-1, vmax=1 
)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.title("Öznitelikler Arası Korelasyon Matrisi (Bildiri Çalışması)")
plt.tight_layout()

plt.savefig("korelasyon_matrisi_duzeltilmis.pdf")


target_corr = df.corr()['burnout_score'].drop('burnout_score')

target_corr = target_corr.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
target_corr.plot(kind='bar', color='skyblue')
plt.title("Burnout Score ile Öznitelikler Arasındaki Korelasyon Katsayıları")
plt.ylabel("Korelasyon Değeri")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
