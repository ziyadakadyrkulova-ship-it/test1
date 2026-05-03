"""
=============================================================
  НЕДЕЛЯ 12 — Разность средних и фильтрация выбросов (Pandas)
=============================================================

Задание (п.12):
  На той же таблице — одно число «разность средних»
  (среднее A – среднее B).
  Опционально: отфильтровать выбросы простым правилом
  (в рамках того же файла).

Что нового в этой неделе:
  Добавляется класс MeanDiffAnalyzer.
  Выбросы определяем методом IQR (межквартильный размах):
  значения ниже Q1 − 1.5·IQR или выше Q3 + 1.5·IQR считаются
  выбросами и удаляются перед пересчётом.
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
        self.groups   = np.array(group_list)
        self.scores   = np.array(score_list)
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
        all_scores = np.concatenate([self.scores_a, self.scores_b])
        self.bin_edges = np.linspace(np.min(all_scores),
                                     np.max(all_scores), self.n_bins + 1)
        self.hist_a, _ = np.histogram(self.scores_a, bins=self.bin_edges)
        self.hist_b, _ = np.histogram(self.scores_b, bins=self.bin_edges)
        print(f"\n[HistogramAnalyzer] Диапазон: "
              f"{np.min(all_scores):.1f}–{np.max(all_scores):.1f}, "
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
#  КОД ИЗ НЕДЕЛИ 11
# ══════════════════════════════════════════════════════════════════

class GroupStatsCalculator:
    """(Неделя 11) Pandas groupby → mean/std/count → group_stats.csv."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df       = pd.read_csv(filepath)
        self.stats_df = None

    def compute(self):
        self.stats_df = (
            self.df.groupby('group')['score']
            .agg(['mean', 'std', 'count'])
            .reset_index()
        )
        self.stats_df.columns = ['group', 'mean', 'std', 'count']
        self.stats_df['mean'] = self.stats_df['mean'].round(2)
        self.stats_df['std']  = self.stats_df['std'].round(2)
        print("\n========== СТАТИСТИКА (Pandas, Неделя 11) ==========")
        print(self.stats_df.to_string(index=False))
        print("=====================================================\n")
        return self.stats_df

    def save(self, out_path='group_stats.csv'):
        if self.stats_df is None:
            self.compute()
        self.stats_df.to_csv(out_path, index=False, encoding='utf-8')
        print(f"[GroupStatsCalculator] Сохранено → {out_path}")


# ══════════════════════════════════════════════════════════════════
#  НОВЫЙ КОД — НЕДЕЛЯ 12
# ══════════════════════════════════════════════════════════════════

class MeanDiffAnalyzer:
    """
    (Неделя 12) Вычисляет разность средних между группами A и B.
    Опционально — фильтрует выбросы по методу IQR перед пересчётом.

    Метод IQR (межквартильный размах):
        Q1 = 25-й процентиль, Q3 = 75-й процентиль
        IQR = Q3 - Q1
        Выбросы: score < Q1 - 1.5*IQR  или  score > Q3 + 1.5*IQR

    Атрибуты:
        stats_df     (pd.DataFrame) — таблица из GroupStatsCalculator
        mean_diff    (float)        — разность средних (A − B)
        filtered_df  (pd.DataFrame) — данные после удаления выбросов
    """

    def __init__(self, stats_df: pd.DataFrame, raw_df: pd.DataFrame):
        """
        :param stats_df: DataFrame из GroupStatsCalculator.compute()
        :param raw_df:   исходный DataFrame (group, score)
        """
        self.stats_df    = stats_df
        self.raw_df      = raw_df
        self.mean_diff   = None
        self.filtered_df = None

    def compute_diff(self) -> float:
        """
        Считает разность средних: mean(A) − mean(B).

        :return: float — разность средних
        """
        mean_a = self.stats_df.loc[
            self.stats_df['group'] == 'A', 'mean'].values[0]
        mean_b = self.stats_df.loc[
            self.stats_df['group'] == 'B', 'mean'].values[0]

        self.mean_diff = round(mean_a - mean_b, 2)

        print("\n========== РАЗНОСТЬ СРЕДНИХ (Неделя 12) ==========")
        print(f"  Среднее группы A : {mean_a}")
        print(f"  Среднее группы B : {mean_b}")
        print(f"  Разность A − B   : {self.mean_diff}")
        print("===================================================\n")

        return self.mean_diff

    def remove_outliers(self) -> pd.DataFrame:
        """
        Удаляет выбросы из каждой группы по правилу IQR.
        Работает внутри каждой группы отдельно — чтобы не «съедать»
        реальные различия между группами.

        :return: отфильтрованный DataFrame
        """
        frames = []
        for group_name, group_df in self.raw_df.groupby('group'):
            scores = group_df['score']
            q1 = scores.quantile(0.25)
            q3 = scores.quantile(0.75)
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr

            filtered = group_df[(scores >= lo) & (scores <= hi)]
            n_removed = len(group_df) - len(filtered)

            print(f"[Outliers] Группа {group_name}: "
                  f"IQR={iqr:.2f}, границы=[{lo:.2f}, {hi:.2f}], "
                  f"удалено выбросов: {n_removed}")
            frames.append(filtered)

        self.filtered_df = pd.concat(frames).reset_index(drop=True)

        print(f"\nСтрок до фильтрации:  {len(self.raw_df)}")
        print(f"Строк после фильтрации: {len(self.filtered_df)}\n")

        # Пересчитываем разность средних на очищенных данных
        new_means = (
            self.filtered_df.groupby('group')['score'].mean().round(2)
        )
        new_diff = round(new_means['A'] - new_means['B'], 2)
        print(f"Разность средних после фильтрации: {new_diff}")

        return self.filtered_df


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

    # --- Неделя 12 ---
    diff_analyzer = MeanDiffAnalyzer(
        stats_df=calc.stats_df,
        raw_df=calc.df
    )
    diff = diff_analyzer.compute_diff()
    print(f"Итоговая разность средних: {diff}")

    filtered = diff_analyzer.remove_outliers()
    print("\nОчищенные данные (первые строки):")
    print(filtered.head(8))