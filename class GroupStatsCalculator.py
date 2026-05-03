"""
=============================================================
  НЕДЕЛЯ 11 — Pandas groupby и сохранение group_stats.csv
=============================================================

Задание (п.11):
  Pandas: groupby('group')['score'].agg(['mean', 'std', 'count'])
  Сохранить результат в group_stats.csv.

Что нового в этой неделе:
  Добавляется класс GroupStatsCalculator.
  Впервые используется Pandas (прошли на 11-й неделе).

Подход:
  Читаем CSV через pd.read_csv.
  groupby + agg для подсчёта статистик.
  Результат сохраняется в group_stats.csv.
"""

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
#  КОД ИЗ НЕДЕЛИ 9
# ══════════════════════════════════════════════════════════════════

class ScoreLoader:
    """(Неделя 9) Загрузка CSV → массивы NumPy."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.groups = self.scores = self.scores_a = self.scores_b = None
        self._load()

    def _load(self):
        group_list, score_list = [], []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                group_list.append(parts[0].strip())
                score_list.append(float(parts[1].strip()))
        self.groups  = np.array(group_list)
        self.scores  = np.array(score_list)
        self.scores_a = self.scores[self.groups == 'A']
        self.scores_b = self.scores[self.groups == 'B']

    def compute_stats(self):
        mean_a, std_a = np.mean(self.scores_a), np.std(self.scores_a)
        mean_b, std_b = np.mean(self.scores_b), np.std(self.scores_b)
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
#  КОД ИЗ НЕДЕЛИ 10
# ══════════════════════════════════════════════════════════════════

class HistogramAnalyzer:
    """(Неделя 10) Гистограммы с общими границами бинов."""

    def __init__(self, scores_a, scores_b, n_bins=10):
        self.scores_a = scores_a
        self.scores_b = scores_b
        self.n_bins   = n_bins
        self.bin_edges = self.hist_a = self.hist_b = None

    def compute(self):
        all_scores    = np.concatenate([self.scores_a, self.scores_b])
        self.bin_edges = np.linspace(np.min(all_scores),
                                     np.max(all_scores), self.n_bins + 1)
        self.hist_a, _ = np.histogram(self.scores_a, bins=self.bin_edges)
        self.hist_b, _ = np.histogram(self.scores_b, bins=self.bin_edges)
        print(f"\n[HistogramAnalyzer] Диапазон: "
              f"{np.min(all_scores):.1f} — {np.max(all_scores):.1f}, "
              f"бинов: {self.n_bins}")
        return {'hist_a': self.hist_a, 'hist_b': self.hist_b,
                'bin_edges': self.bin_edges}

    def print_histograms(self):
        if self.hist_a is None:
            self.compute()
        print("\n========== ГИСТОГРАММЫ (Неделя 10) ==========")
        for i in range(len(self.hist_a)):
            lo, hi = self.bin_edges[i], self.bin_edges[i + 1]
            print(f"[{lo:5.1f}–{hi:5.1f}]  "
                  f"A:{self.hist_a[i]:2d} {'█'*int(self.hist_a[i])}  "
                  f"B:{self.hist_b[i]:2d} {'█'*int(self.hist_b[i])}")
        print("==============================================\n")


# ══════════════════════════════════════════════════════════════════
#  НОВЫЙ КОД — НЕДЕЛЯ 11
# ══════════════════════════════════════════════════════════════════

class GroupStatsCalculator:
    """
    (Неделя 11) Использует Pandas для вычисления агрегированной
    статистики по группам и сохраняет результат в CSV.

    Почему Pandas удобнее NumPy здесь?
    - groupby + agg — одна строка вместо масок и циклов.
    - Результат — готовый DataFrame, который легко сохранить в CSV.

    Атрибуты:
        filepath    (str)           — путь к исходному CSV
        df          (pd.DataFrame)  — загруженные данные
        stats_df    (pd.DataFrame)  — таблица со статистикой по группам
    """

    def __init__(self, filepath: str):
        """
        :param filepath: путь к scores_groups.csv
        """
        self.filepath = filepath
        self.df       = None
        self.stats_df = None
        self._load()

    def _load(self):
        """Загружает CSV в DataFrame."""
        self.df = pd.read_csv(self.filepath)
        print(f"\n[GroupStatsCalculator] Загружено {len(self.df)} строк.")
        print(self.df.head())

    def compute(self):
        """
        Группирует по колонке 'group', агрегирует 'score':
          - mean  — среднее значение
          - std   — стандартное отклонение (ddof=1, выборочное)
          - count — количество наблюдений

        :return: pd.DataFrame со статистикой
        """
        self.stats_df = (
            self.df
            .groupby('group')['score']
            .agg(['mean', 'std', 'count'])
            .reset_index()                       # group станет обычной колонкой
        )
        # Переименуем для читаемости
        self.stats_df.columns = ['group', 'mean', 'std', 'count']

        # Округляем для красивого вывода
        self.stats_df['mean'] = self.stats_df['mean'].round(2)
        self.stats_df['std']  = self.stats_df['std'].round(2)

        print("\n========== СТАТИСТИКА (Pandas, Неделя 11) ==========")
        print(self.stats_df.to_string(index=False))
        print("=====================================================\n")

        return self.stats_df

    def save(self, out_path: str = 'group_stats.csv'):
        """
        Сохраняет таблицу со статистикой в CSV-файл.

        :param out_path: путь для сохранения (по умолчанию group_stats.csv)
        """
        if self.stats_df is None:
            self.compute()
        self.stats_df.to_csv(out_path, index=False, encoding='utf-8')
        print(f"[GroupStatsCalculator] Статистика сохранена → {out_path}")


# ------------------------------------------------------------------
# Точка входа
# ------------------------------------------------------------------
if __name__ == '__main__':
    # --- Неделя 9 ---
    loader = ScoreLoader('scores_groups.csv')
    loader.compute_stats()

    # --- Неделя 10 ---
    analyzer = HistogramAnalyzer(loader.scores_a, loader.scores_b, n_bins=8)
    analyzer.compute()
    analyzer.print_histograms()

    # --- Неделя 11 ---
    calc = GroupStatsCalculator('scores_groups.csv')
    calc.compute()
    calc.save('group_stats.csv')

    # Проверяем сохранённый файл
    print("\nСодержимое group_stats.csv:")
    print(pd.read_csv('group_stats.csv'))

# Загружаем CSV в DataFrame
df = pd.read_csv('scores_groups.csv')

# groupby('group') — группируем по колонке group
# ['score'] — берём колонку score
# .agg([...]) — считаем сразу несколько метрик
group_stats = df.groupby('group')['score'].agg(['mean', 'std', 'count'])

print(group_stats)

# Сохраняем результат в новый CSV файл
group_stats.to_csv('group_stats.csv')
print("\nФайл group_stats.csv сохранён!")