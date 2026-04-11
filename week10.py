import numpy as np  # подключаем NumPy для работы с массивами

# Создаём пустые списки для хранения данных из файла
scores_all = []
groups_all = []

# Открываем CSV файл для чтения
with open('scores_groups.csv', 'r') as f:
    header = f.readline()  # пропускаем первую строку (заголовок: group,score)
    for line in f:         # проходим по каждой строке
        parts = line.strip().split(',')  # убираем пробелы, делим по запятой
        groups_all.append(parts[0])         # parts[0] — группа (A или B)
        scores_all.append(float(parts[1]))  # parts[1] — балл, переводим в число

# Преобразуем списки в NumPy-массивы
scores = np.array(scores_all)  # массив чисел (баллы)
groups = np.array(groups_all)  # массив строк (метки групп)

# Булевы маски — True там где нужная группа, False везде остальном
mask_A = groups == 'A'
mask_B = groups == 'B'

# Применяем маски — получаем баллы только нужной группы
scores_A = scores[mask_A]  # баллы группы A
scores_B = scores[mask_B]  # баллы группы B

# Считаем среднее и стандартное отклонение
mean_A = np.mean(scores_A)  # среднее группы A
std_A  = np.std(scores_A)   # стандартное отклонение группы A

mean_B = np.mean(scores_B)  # среднее группы B
std_B  = np.std(scores_B)   # стандартное отклонение группы B

# Выводим результаты задачи 9
print(f"Группа A: n={len(scores_A)}")
print(f"  Среднее      : {mean_A:.2f}")
print(f"  Станд. откл. : {std_A:.2f}")

print(f"\nГруппа B: n={len(scores_B)}")
print(f"  Среднее      : {mean_B:.2f}")
print(f"  Станд. откл. : {std_B:.2f}")

print(f"\nРазница средних (B - A): {mean_B - mean_A:.2f}")

# ── Задача 10 ─────────────────────────────────────────────────

# Объединяем оба массива в один чтобы найти общий диапазон
# Это важно — бины должны быть одинаковые для A и B, иначе сравнение некорректно
all_scores = np.concatenate([scores_A, scores_B])

# Находим минимум и максимум по всем данным
global_min = np.min(all_scores)  # самый маленький балл из всех
global_max = np.max(all_scores)  # самый большой балл из всех

# Создаём 11 точек от min до max — получаем 10 равных интервалов (бинов)
bins = np.linspace(global_min, global_max, 11)

# np.histogram считает сколько значений попало в каждый бин
# возвращает: counts (частоты) и edges (границы бинов)
counts_A, edges = np.histogram(scores_A, bins=bins)  # гистограмма группы A
counts_B, edges = np.histogram(scores_B, bins=bins)  # гистограмма группы B

# Выводим результаты задачи 10
print(f"\nГраницы бинов : {np.round(edges, 2)}")  # округляем до 2 знаков
print(f"\nГруппа A — частоты: {counts_A}")  # сколько значений в каждом интервале
print(f"Группа B — частоты: {counts_B}")