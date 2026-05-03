"""
=============================================================
  НЕДЕЛЯ 13 — Визуализация: гистограммы и boxplot (Matplotlib)
=============================================================

Задание (п.13):
  Две гистограммы score рядом (subplot) с общей шкалой по оси X
  ИЛИ boxplot по group. Сохранить как PNG.

Что нового в этой неделе:
  Добавляется класс PlotBuilder.
  Используется matplotlib.pyplot.
  Строятся два subplot: слева — гистограммы, справа — boxplot.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # без GUI — сохраняем в файл
import matplotlib.pyplot as plt


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
#  КОД ИЗ НЕДЕЛИ 12
# ══════════════════════════════════════════════════════════════════

class MeanDiffAnalyzer:
    """(Неделя 12) Разность средних + фильтрация выбросов IQR."""

    def __init__(self, stats_df, raw_df):
        self.stats_df    = stats_df
        self.raw_df      = raw_df
        self.mean_diff   = None
        self.filtered_df = None

    def compute_diff(self):
        mean_a = self.stats_df.loc[self.stats_df['group']=='A','mean'].values[0]
        mean_b = self.stats_df.loc[self.stats_df['group']=='B','mean'].values[0]
        self.mean_diff = round(mean_a - mean_b, 2)
        print("\n========== РАЗНОСТЬ СРЕДНИХ (Неделя 12) ==========")
        print(f"  Среднее A : {mean_a} | Среднее B : {mean_b}")
        print(f"  Разность A − B : {self.mean_diff}")
        print("===================================================\n")
        return self.mean_diff

    def remove_outliers(self):
        frames = []
        for name, grp in self.raw_df.groupby('group'):
            q1, q3 = grp['score'].quantile(0.25), grp['score'].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
            filtered = grp[(grp['score'] >= lo) & (grp['score'] <= hi)]
            print(f"[Outliers] Группа {name}: удалено {len(grp)-len(filtered)}")
            frames.append(filtered)
        self.filtered_df = pd.concat(frames).reset_index(drop=True)
        return self.filtered_df


# ══════════════════════════════════════════════════════════════════
#  НОВЫЙ КОД — НЕДЕЛЯ 13
# ══════════════════════════════════════════════════════════════════

class PlotBuilder:
    """
    (Неделя 13) Строит и сохраняет графики в PNG.

    График состоит из двух subplot:
      Левый  — две гистограммы с общей шкалой X (для сравнения).
      Правый — boxplot по group (показывает медиану, IQR, выбросы).

    Атрибуты:
        scores_a  (np.ndarray) — баллы группы A
        scores_b  (np.ndarray) — баллы группы B
        n_bins    (int)        — число бинов гистограммы
        out_path  (str)        — путь сохранения PNG
    """

    def __init__(self, scores_a: np.ndarray, scores_b: np.ndarray,
                 n_bins: int = 10, out_path: str = 'score_plots.png'):
        self.scores_a = scores_a
        self.scores_b = scores_b
        self.n_bins   = n_bins
        self.out_path = out_path

    def _get_common_bins(self):
        """Вычисляет общие границы бинов для двух групп."""
        all_scores = np.concatenate([self.scores_a, self.scores_b])
        return np.linspace(np.min(all_scores),
                           np.max(all_scores), self.n_bins + 1)

    def build_and_save(self):
        """
        Создаёт фигуру с двумя subplot и сохраняет в PNG.

        Шаги:
          1. Создаём fig, (ax1, ax2) через plt.subplots(1, 2).
          2. На ax1 рисуем две гистограммы с alpha=0.6 (полупрозрачность)
             и общими bin_edges.
          3. На ax2 рисуем boxplot для обеих групп.
          4. Добавляем заголовки, подписи осей, легенду.
          5. Сохраняем plt.savefig → PNG.
        """
        bin_edges = self._get_common_bins()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Сравнение баллов групп A и B', fontsize=14, y=1.02)

        # ── Левый subplot: гистограммы ──────────────────────────────
        ax1.hist(self.scores_a, bins=bin_edges,
                 alpha=0.6, color='steelblue', label='Группа A',
                 edgecolor='white')
        ax1.hist(self.scores_b, bins=bin_edges,
                 alpha=0.6, color='coral',     label='Группа B',
                 edgecolor='white')

        ax1.set_title('Гистограммы score (общая шкала X)')
        ax1.set_xlabel('Балл')
        ax1.set_ylabel('Количество')
        ax1.legend()

        # Вертикальные линии — средние значений
        ax1.axvline(np.mean(self.scores_a), color='steelblue',
                    linestyle='--', linewidth=1.5,
                    label=f'Среднее A = {np.mean(self.scores_a):.1f}')
        ax1.axvline(np.mean(self.scores_b), color='coral',
                    linestyle='--', linewidth=1.5,
                    label=f'Среднее B = {np.mean(self.scores_b):.1f}')
        ax1.legend(fontsize=8)

        # ── Правый subplot: boxplot ─────────────────────────────────
        bp = ax2.boxplot(
            [self.scores_a, self.scores_b],
            patch_artist=True,          # заливка цветом
            tick_labels =['Группа A', 'Группа B'],
            notch=False,
        )
        # Раскрашиваем ящики
        colors = ['steelblue', 'coral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax2.set_title('Boxplot score по группам')
        ax2.set_ylabel('Балл')

        plt.tight_layout()
        plt.savefig(self.out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n[PlotBuilder] График сохранён → {self.out_path}")


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
    diff_analyzer = MeanDiffAnalyzer(calc.stats_df, calc.df)
    diff_analyzer.compute_diff()
    diff_analyzer.remove_outliers()

    # --- Неделя 13 ---
    plotter = PlotBuilder(
        scores_a=loader.scores_a,
        scores_b=loader.scores_b,
        n_bins=8,
        out_path='score_plots.png'
    )
    plotter.build_and_save()
    print("Готово! Файл score_plots.png создан.")