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

# Выводим результаты
print(f"Группа A: n={len(scores_A)}")
print(f"  Среднее      : {mean_A:.2f}")
print(f"  Станд. откл. : {std_A:.2f}")

print(f"\nГруппа B: n={len(scores_B)}")
print(f"  Среднее      : {mean_B:.2f}")
print(f"  Станд. откл. : {std_B:.2f}")

print(f"\nРазница средних (B - A): {mean_B - mean_A:.2f}")