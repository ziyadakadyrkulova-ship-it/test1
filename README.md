import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# 1. Загрузка данных
df = pd.read_csv("scores_groups.csv")

# Проверим названия колонок
print(df.head())
print(df.columns)

# 2. Выделяем массивы score для групп A и B
scores_A = df[df["group"] == "A"]["score"].to_numpy()
scores_B = df[df["group"] == "B"]["score"].to_numpy()

# 3. Среднее и стандартное отклонение в каждой группе
mean_A = np.mean(scores_A)
std_A = np.std(scores_A, ddof=1)   # выборочное стандартное отклонение
mean_B = np.mean(scores_B)
std_B = np.std(scores_B, ddof=1)

print("Группа A:")
print("Среднее =", mean_A)
print("Ст. отклонение =", std_A)

print("Группа B:")
print("Среднее =", mean_B)
print("Ст. отклонение =", std_B)

# 4. Общие границы для гистограмм
all_scores = np.concatenate([scores_A, scores_B])
min_score = all_scores.min()
max_score = all_scores.max()

# Одинаковые bins для обеих групп
bins = np.linspace(min_score, max_score, 11)  # 10 интервалов

hist_A, _ = np.histogram(scores_A, bins=bins)
hist_B, _ = np.histogram(scores_B, bins=bins)

print("Гистограмма A:", hist_A)
print("Гистограмма B:", hist_B)

# 5. Статистика через pandas groupby
group_stats = df.groupby("group")["score"].agg(["mean", "std", "count"])
group_stats.to_csv("group_stats.csv", index=True)

print("\nТаблица статистик:")
print(group_stats)

# 6. Разность средних
mean_diff = mean_A - mean_B   # фиксируем: среднее A - среднее B
print("\nРазность средних (A - B) =", mean_diff)

# 7. Гистограммы рядом
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(scores_A, bins=bins, alpha=0.7, edgecolor='black')
plt.title("Гистограмма score: группа A")
plt.xlabel("score")
plt.ylabel("Частота")

plt.subplot(1, 2, 2)
plt.hist(scores_B, bins=bins, alpha=0.7, edgecolor='black')
plt.title("Гистограмма score: группа B")
plt.xlabel("score")
plt.ylabel("Частота")

plt.tight_layout()
plt.savefig("group_histograms.png")
plt.show()

# 8. Boxplot по группам
plt.figure(figsize=(6, 5))
df.boxplot(column="score", by="group")
plt.title("Boxplot score по группам")
plt.suptitle("")
plt.xlabel("group")
plt.ylabel("score")
plt.savefig("group_boxplot.png")
plt.show()

# 9. JSON-ответ
result_json = {
    "A": {
        "mean": float(mean_A),
        "std": float(std_A),
        "count": int(len(scores_A))
    },
    "B": {
        "mean": float(mean_B),
        "std": float(std_B),
        "count": int(len(scores_B))
    },
    "mean_difference_A_minus_B": float(mean_diff)
}

with open("group_stats.json", "w", encoding="utf-8") as f:
    json.dump(result_json, f, ensure_ascii=False, indent=4)

print("\nJSON:")
print(json.dumps(result_json, ensure_ascii=False, indent=4))
