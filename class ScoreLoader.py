"""
=============================================================
  НЕДЕЛЯ 9 — Загрузка CSV и базовая статистика (NumPy, ООП)
=============================================================

Задание (п.9):
  Загрузить колонки score и group в массивы NumPy.
  По маскам выделить два массива баллов — только для группы A
  и только для группы B.
  Посчитать среднее и стандартное отклонение в каждой группе.

"""

import numpy as np


class ScoreLoader:
    """
    Класс для загрузки файла scores_groups.csv и
    вычисления базовой статистики по каждой группе.

    Атрибуты:
        filepath  (str)        — путь к CSV-файлу
        groups    (np.ndarray) — массив строк с метками групп
        scores    (np.ndarray) — массив числовых баллов (float)
        scores_a  (np.ndarray) — баллы только группы A
        scores_b  (np.ndarray) — баллы только группы B
    """

    def __init__(self, filepath: str):
        """
        Конструктор: запоминает путь к файлу и сразу
        запускает загрузку данных.

        :param filepath: путь к CSV-файлу (например, 'scores_groups.csv')
        """
        self.filepath = filepath
        self.groups = None    # будет np.ndarray строк
        self.scores = None    # будет np.ndarray float
        self.scores_a = None  # маска группы A
        self.scores_b = None  # маска группы B
        self._load()          # вызываем загрузку сразу

    # ------------------------------------------------------------------
    # Приватный метод загрузки
    # ------------------------------------------------------------------
    def _load(self):
        """
        Читает CSV-файл построчно (без Pandas!).
        Первая строка — заголовок, пропускается.
        Заполняет self.groups и self.scores, затем формирует маски.
        """
        group_list = []
        score_list = []

        with open(self.filepath, 'r', encoding='utf-8') as f:
            header = f.readline()  # пропускаем строку заголовка
            print(f"[ScoreLoader] Заголовок файла: {header.strip()}")

            for line in f:
                line = line.strip()
                if not line:        # пропускаем пустые строки
                    continue
                parts = line.split(',')
                group_list.append(parts[0].strip())
                score_list.append(float(parts[1].strip()))

        # Переводим списки в массивы NumPy
        self.groups = np.array(group_list)   # dtype=str
        self.scores = np.array(score_list)   # dtype=float64

        print(f"[ScoreLoader] Загружено строк: {len(self.scores)}")
        print(f"[ScoreLoader] Уникальные группы: {np.unique(self.groups)}")

        # Маски для фильтрации по группе
        mask_a = (self.groups == 'A')
        mask_b = (self.groups == 'B')

        self.scores_a = self.scores[mask_a]
        self.scores_b = self.scores[mask_b]

    # ------------------------------------------------------------------
    # Публичный метод вычисления статистики
    # ------------------------------------------------------------------
    def compute_stats(self):
        """
        Вычисляет и выводит среднее (mean) и стандартное отклонение (std)
        для каждой группы с помощью NumPy.

        :return: словарь {'A': {'mean': ..., 'std': ...},
                          'B': {'mean': ..., 'std': ...}}
        """
        mean_a = np.mean(self.scores_a)
        std_a  = np.std(self.scores_a)   # std по генеральной совокупности

        mean_b = np.mean(self.scores_b)
        std_b  = np.std(self.scores_b)

        print("\n========== СТАТИСТИКА ПО ГРУППАМ (Неделя 9) ==========")
        print(f"Группа A | Кол-во: {len(self.scores_a):3d} | "
              f"Среднее: {mean_a:.2f} | Стд. откл.: {std_a:.2f}")
        print(f"Группа B | Кол-во: {len(self.scores_b):3d} | "
              f"Среднее: {mean_b:.2f} | Стд. откл.: {std_b:.2f}")
        print("=======================================================\n")

        return {
            'A': {'mean': mean_a, 'std': std_a, 'count': len(self.scores_a)},
            'B': {'mean': mean_b, 'std': std_b, 'count': len(self.scores_b)},
        }


# ------------------------------------------------------------------
# Точка входа
# ------------------------------------------------------------------
if __name__ == '__main__':
    loader = ScoreLoader('scores_groups.csv')
    stats = loader.compute_stats()

    # Дополнительный вывод для наглядности
    print("Первые 5 баллов группы A:", loader.scores_a[:5])
    print("Первые 5 баллов группы B:", loader.scores_b[:5])
