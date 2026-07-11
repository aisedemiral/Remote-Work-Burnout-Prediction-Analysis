import matplotlib.pyplot as plt
import seaborn as sns

models = {
    "Logistic Regression": y_pred_lr,
    "SVM": y_pred_svm,
    "Random Forest": y_pred_rf,
    "Gradient Boosting": y_pred_gb
}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, (name, pred) in enumerate(models.items()):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues', cbar=False)
    axes[i].set_title(name)
    axes[i].set_xlabel("Tahmin")
    axes[i].set_ylabel("Gerçek")

plt.tight_layout()
plt.savefig("tum_modeller_confusion_matrix.pdf")
plt.show()
