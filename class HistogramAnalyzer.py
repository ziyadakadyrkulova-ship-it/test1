"""
=============================================================
  НЕДЕЛЯ 10 — Гистограммы с NumPy (ООП, без Pandas)
=============================================================

Задание (п.10):
  На тех же двух массивах — вычислить общие границы для гистограмм:
  min/max по объединённым данным, затем np.histogram для каждой
  группы с одинаковыми bins (или одинаковое число бинов).

Что нового в этой неделе:
  Добавляется класс HistogramAnalyzer, который принимает уже
  загруженные массивы scores_a и scores_b и строит гистограммы
  с общими границами бинов.

Подход:
  Pandas НЕ используется.
  Используются numpy: np.concatenate, np.histogram, np.linspace.
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════
#  КОД ИЗ НЕДЕЛИ 9 (оставляем, чтобы всё работало в одном файле)
# ══════════════════════════════════════════════════════════════════

class ScoreLoader:
    """
    (Неделя 9) Загружает CSV и формирует массивы NumPy.
    Подробные комментарии — см. week_09.py.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.groups = None
        self.scores = None
        self.scores_a = None
        self.scores_b = None
        self._load()

    def _load(self):
        group_list, score_list = [], []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            f.readline()  # заголовок
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                group_list.append(parts[0].strip())
                score_list.append(float(parts[1].strip()))

        self.groups = np.array(group_list)
        self.scores = np.array(score_list)

        mask_a = (self.groups == 'A')
        mask_b = (self.groups == 'B')
        self.scores_a = self.scores[mask_a]
        self.scores_b = self.scores[mask_b]

    def compute_stats(self):
        mean_a = np.mean(self.scores_a)
        std_a  = np.std(self.scores_a)
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


# ══════════════════════════════════════════════════════════════════
#  НОВЫЙ КОД — НЕДЕЛЯ 10
# ══════════════════════════════════════════════════════════════════

class HistogramAnalyzer:
    """
    (Неделя 10) Строит гистограммы для двух групп с общими границами бинов.

    Зачем общие границы?
    Если каждая группа строит гистограмму по своему диапазону,
    сравнивать их некорректно — шкала X будет разная.
    Мы объединяем данные, берём общий min и max, и делим
    этот диапазон на одинаковые бины для обеих групп.

    Атрибуты:
        scores_a   (np.ndarray) — баллы группы A
        scores_b   (np.ndarray) — баллы группы B
        n_bins     (int)        — количество бинов (по умолчанию 10)
        bin_edges  (np.ndarray) — общие границы бинов
        hist_a     (np.ndarray) — частоты группы A
        hist_b     (np.ndarray) — частоты группы B
    """

    def __init__(self, scores_a: np.ndarray, scores_b: np.ndarray,
                 n_bins: int = 10):
        """
        :param scores_a: массив баллов группы A
        :param scores_b: массив баллов группы B
        :param n_bins:   число бинов для гистограммы
        """
        self.scores_a = scores_a
        self.scores_b = scores_b
        self.n_bins   = n_bins
        self.bin_edges = None
        self.hist_a    = None
        self.hist_b    = None

    def compute(self):
        """
        Шаг 1: объединяем данные двух групп → общий min и max.
        Шаг 2: строим равномерные границы бинов (np.linspace).
        Шаг 3: считаем np.histogram для каждой группы с этими границами.

        :return: словарь с hist_a, hist_b и bin_edges
        """
        # Объединяем все баллы для поиска общих границ
        all_scores = np.concatenate([self.scores_a, self.scores_b])

        global_min = np.min(all_scores)
        global_max = np.max(all_scores)

        print(f"\n[HistogramAnalyzer] Общий диапазон баллов: "
              f"{global_min:.1f} — {global_max:.1f}")

        # n_bins + 1 точек → n_bins интервалов
        self.bin_edges = np.linspace(global_min, global_max, self.n_bins + 1)

        # np.histogram возвращает (частоты, границы)
        self.hist_a, _ = np.histogram(self.scores_a, bins=self.bin_edges)
        self.hist_b, _ = np.histogram(self.scores_b, bins=self.bin_edges)

        return {
            'hist_a':     self.hist_a,
            'hist_b':     self.hist_b,
            'bin_edges':  self.bin_edges,
        }

    def print_histograms(self):
        """
        Выводит гистограммы в текстовом виде (ASCII-арт)
        для наглядности без matplotlib.
        """
        if self.hist_a is None:
            self.compute()

        print("\n========== ГИСТОГРАММЫ (Неделя 10) ==========")
        print(f"{'Бин':>20}  {'Группа A':>10}  {'Группа B':>10}")
        print("-" * 46)
        for i in range(len(self.hist_a)):
            lo = self.bin_edges[i]
            hi = self.bin_edges[i + 1]
            bar_a = '█' * int(self.hist_a[i])
            bar_b = '█' * int(self.hist_b[i])
            print(f"[{lo:5.1f} – {hi:5.1f}]  "
                  f"A:{self.hist_a[i]:2d} {bar_a}")
            print(f"{'':20}  "
                  f"B:{self.hist_b[i]:2d} {bar_b}")
        print("==============================================\n")


# ------------------------------------------------------------------
# Точка входа
# ------------------------------------------------------------------
if __name__ == '__main__':
    # --- Неделя 9: загрузка и статистика ---
    loader = ScoreLoader('scores_groups.csv')
    stats = loader.compute_stats()

    # --- Неделя 10: гистограммы ---
    analyzer = HistogramAnalyzer(loader.scores_a, loader.scores_b, n_bins=8)
    result = analyzer.compute()
    analyzer.print_histograms()

    print("Границы бинов:", np.round(result['bin_edges'], 2))
    print("Частоты A:    ", result['hist_a'])
    print("Частоты B:    ", result['hist_b'])