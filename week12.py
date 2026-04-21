import numpy as np
import pandas as pd

# ── Задачи 9-11 (база) ───────────────────────────────────────
scores_all = []
groups_all = []

with open('scores_groups.csv', 'r') as f:
    header = f.readline()
    for line in f:
        parts = line.strip().split(',')
        groups_all.append(parts[0])
        scores_all.append(float(parts[1]))

scores = np.array(scores_all)
groups = np.array(groups_all)

mask_A = groups == 'A'
mask_B = groups == 'B'

scores_A = scores[mask_A]
scores_B = scores[mask_B]

mean_A = np.mean(scores_A)
std_A  = np.std(scores_A)
mean_B = np.mean(scores_B)
std_B  = np.std(scores_B)

print(f"Группа A: n={len(scores_A)}, Среднее: {mean_A:.2f}, Станд. откл.: {std_A:.2f}")
print(f"Группа B: n={len(scores_B)}, Среднее: {mean_B:.2f}, Станд. откл.: {std_B:.2f}")
print(f"Разница средних (B - A): {mean_B - mean_A:.2f}")

all_scores = np.concatenate([scores_A, scores_B])
global_min = np.min(all_scores)
global_max = np.max(all_scores)
bins = np.linspace(global_min, global_max, 11)
counts_A, edges = np.histogram(scores_A, bins=bins)
counts_B, edges = np.histogram(scores_B, bins=bins)
print(f"\nГраницы бинов : {np.round(edges, 2)}")
print(f"Группа A — частоты: {counts_A}")
print(f"Группа B — частоты: {counts_B}")

df = pd.read_csv('scores_groups.csv')
group_stats = df.groupby('group')['score'].agg(['mean', 'std', 'count'])
print(f"\n{group_stats}")
group_stats.to_csv('group_stats.csv')

# ── Задача 12 ─────────────────────────────────────────────────
print("\n--- Задача 12: разность средних + выбросы ---")

# Разность средних из таблицы groupby
mean_A_pd = group_stats.loc['A', 'mean']  # среднее группы A из pandas
mean_B_pd = group_stats.loc['B', 'mean']  # среднее группы B из pandas
diff = mean_A_pd - mean_B_pd              # разность средних (A - B)
print(f"Разность средних (A - B): {diff:.2f}")

# Фильтрация выбросов простым правилом: убираем значения дальше 3*std от среднего
overall_mean = np.mean(all_scores)  # общее среднее
overall_std  = np.std(all_scores)   # общее стандартное отклонение

# Маска: оставляем только те значения, которые в пределах 3 стандартных отклонений
no_outliers = df[np.abs(df['score'] - overall_mean) <= 3 * overall_std]

print(f"\nДо фильтрации выбросов : {len(df)} строк")
print(f"После фильтрации выбросов: {len(no_outliers)} строк")

# Сохраняем очищенные данные
no_outliers.to_csv('group_stats.csv', index=False)
print("Файл group_stats.csv обновлён (без выбросов)!")