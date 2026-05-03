import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#  КЛАСС 1 — ScoreLoader  (Неделя 9)
"""
  НЕДЕЛЯ 9 — Загрузка CSV и базовая статистика (NumPy, ООП)
Задание (п.9):
  Загрузить колонки score и group в массивы NumPy.
  По маскам выделить два массива баллов — только для группы A
  и только для группы B.
  Посчитать среднее и стандартное отклонение в каждой группе.

"""
class ScoreLoader:
    """
    Неделя 9 — Загрузка CSV и базовая статистика (NumPy).

    ЗАЧЕМ НУЖЕН ЭТОТ КЛАСС:
      Является фундаментом всей программы. Читает CSV вручную
      (без Pandas), формирует массивы NumPy и сразу разделяет
      данные на две группы по маске.

    АТРИБУТЫ:
      filepath  (str)        — путь к CSV
      groups    (np.ndarray) — массив строк с метками ['A','B',...]
      scores    (np.ndarray) — массив баллов в виде float64
      scores_a  (np.ndarray) — только баллы группы A
      scores_b  (np.ndarray) — только баллы группы B

    МЕТОДЫ:
      _load()         — приватный, вызывается в __init__
      compute_stats() — публичный, возвращает словарь статистик
    """

    def __init__(self, filepath: str):
        """
        Инициализация: запомнить путь и немедленно загрузить данные.

         filepath: путь к файлу, например 'scores_groups.csv'
        """
        self.filepath = filepath
        self.groups   = None
        self.scores   = None
        self.scores_a = None
        self.scores_b = None
        self._load()   # загружаем сразу при создании объекта

    def _load(self):
        """
        Читает CSV построчно без Pandas.
        Первая строка — заголовок, пропускается через readline().
        Каждая следующая строка разбивается по запятой.
        Списки конвертируются в numpy-массивы.
        Маски (==) выделяют каждую группу.
        """
        group_list, score_list = [], []

        with open(self.filepath, 'r', encoding='utf-8') as f:
            f.readline()           # пропускаем заголовок 'group,score'
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                group_list.append(parts[0].strip())
                score_list.append(float(parts[1].strip()))

        # Переводим в массивы NumPy
        self.groups = np.array(group_list)    # строковый массив
        self.scores = np.array(score_list)    # числовой массив float64

        # Булевые маски → подмассивы
        self.scores_a = self.scores[self.groups == 'A']
        self.scores_b = self.scores[self.groups == 'B']

        print(f"[ScoreLoader] Загружено: {len(self.scores)} строк | "
              f"A={len(self.scores_a)}, B={len(self.scores_b)}")

    def compute_stats(self) -> dict:
        """
        Вычисляет среднее (np.mean) и стандартное отклонение (np.std)
        для каждой группы.

        std — по генеральной совокупности (ddof=0),
        т.к. мы рассматриваем имеющиеся данные как всю совокупность.

        :return: {'A': {'mean': float, 'std': float, 'count': int},
                  'B': {'mean': float, 'std': float, 'count': int}}
        """
        mean_a = float(np.mean(self.scores_a))
        std_a  = float(np.std(self.scores_a))
        mean_b = float(np.mean(self.scores_b))
        std_b  = float(np.std(self.scores_b))

        print("\n" + "═"*54)
        print("  СТАТИСТИКА ПО ГРУППАМ  (Неделя 9)")
        print("═"*54)
        print(f"  Группа A | n={len(self.scores_a):3d} | "
              f"Среднее={mean_a:.2f} | Стд.откл.={std_a:.2f}")
        print(f"  Группа B | n={len(self.scores_b):3d} | "
              f"Среднее={mean_b:.2f} | Стд.откл.={std_b:.2f}")
        print("═"*54 + "\n")

        return {
            'A': {'mean': mean_a, 'std': std_a, 'count': len(self.scores_a)},
            'B': {'mean': mean_b, 'std': std_b, 'count': len(self.scores_b)},
        }


#  КЛАСС 2 — HistogramAnalyzer  (Неделя 10)

class HistogramAnalyzer:
    """
    Неделя 10 — Гистограммы с общими границами бинов (NumPy).

    ЗАЧЕМ НУЖЕН ЭТОТ КЛАСС:
      Чтобы сравнение гистограмм было честным, обе группы должны
      использовать одинаковые границы бинов. Класс находит
      глобальный min/max по объединённым данным, строит
      равномерную сетку (np.linspace) и применяет np.histogram
      к каждой группе отдельно.

    АТРИБУТЫ:
      scores_a   (np.ndarray) — баллы группы A
      scores_b   (np.ndarray) — баллы группы B
      n_bins     (int)        — число бинов
      bin_edges  (np.ndarray) — n_bins+1 граней
      hist_a     (np.ndarray) — частоты группы A по бинам
      hist_b     (np.ndarray) — частоты группы B по бинам
    """

    def __init__(self, scores_a: np.ndarray, scores_b: np.ndarray,
                 n_bins: int = 10):
        self.scores_a  = scores_a
        self.scores_b  = scores_b
        self.n_bins    = n_bins
        self.bin_edges = None
        self.hist_a    = None
        self.hist_b    = None

    def compute(self) -> dict:
        """
        1. Объединяем данные → global_min, global_max.
        2. np.linspace(min, max, n_bins+1) → bin_edges.
        3. np.histogram(scores_X, bins=bin_edges) → hist_X.

        :return: {'hist_a', 'hist_b', 'bin_edges'}
        """
        all_scores     = np.concatenate([self.scores_a, self.scores_b])
        global_min     = np.min(all_scores)
        global_max     = np.max(all_scores)

        # n_bins интервалов → нужно n_bins+1 граней
        self.bin_edges = np.linspace(global_min, global_max, self.n_bins + 1)

        self.hist_a, _ = np.histogram(self.scores_a, bins=self.bin_edges)
        self.hist_b, _ = np.histogram(self.scores_b, bins=self.bin_edges)

        print(f"[HistogramAnalyzer] Диапазон [{global_min:.1f}, {global_max:.1f}], "
              f"бинов: {self.n_bins}")

        return {'hist_a': self.hist_a, 'hist_b': self.hist_b,
                'bin_edges': self.bin_edges}

    def print_histograms(self):
        """Выводит ASCII-гистограммы в консоль."""
        if self.hist_a is None:
            self.compute()

        print("\n" + "═"*54)
        print("  ГИСТОГРАММЫ  (Неделя 10)")
        print("═"*54)
        for i in range(len(self.hist_a)):
            lo = self.bin_edges[i]
            hi = self.bin_edges[i + 1]
            print(f"  [{lo:5.1f}–{hi:5.1f}]  "
                  f"A:{self.hist_a[i]:2d} {'▓'*int(self.hist_a[i])}  "
                  f"B:{self.hist_b[i]:2d} {'░'*int(self.hist_b[i])}")
        print("═"*54 + "\n")


# ══════════════════════════════════════════════════════════════════
#  КЛАСС 3 — GroupStatsCalculator  (Неделя 11)
# ══════════════════════════════════════════════════════════════════

class GroupStatsCalculator:
    """
    Неделя 11 — Pandas groupby/agg и сохранение в group_stats.csv.

    ЗАЧЕМ НУЖЕН ЭТОТ КЛАСС:
      На 11-й неделе прошли Pandas. groupby + agg позволяют
      получить сводную таблицу статистик одной цепочкой вызовов.
      Результат легко сохранить в CSV через to_csv().

    АТРИБУТЫ:
      filepath  (str)          — путь к исходному CSV
      df        (pd.DataFrame) — полные данные
      stats_df  (pd.DataFrame) — сводная таблица (group/mean/std/count)
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df       = pd.read_csv(filepath)   # Pandas читает CSV
        self.stats_df = None
        print(f"[GroupStatsCalculator] Загружено {len(self.df)} строк через Pandas.")

    def compute(self) -> pd.DataFrame:
        """
        groupby('group')['score'].agg(['mean','std','count'])
        reset_index() — переводит индекс-group в обычную колонку.
        round(2)      — округление для читаемости.

        :return: pd.DataFrame с колонками [group, mean, std, count]
        """
        self.stats_df = (
            self.df
            .groupby('group')['score']
            .agg(['mean', 'std', 'count'])
            .reset_index()
        )
        self.stats_df.columns = ['group', 'mean', 'std', 'count']
        self.stats_df['mean'] = self.stats_df['mean'].round(2)
        self.stats_df['std']  = self.stats_df['std'].round(2)

        print("\n" + "═"*54)
        print("  ГРУППОВАЯ СТАТИСТИКА PANDAS  (Неделя 11)")
        print("═"*54)
        print(self.stats_df.to_string(index=False))
        print("═"*54 + "\n")
        return self.stats_df

    def save(self, out_path: str = 'group_stats.csv'):
        """
        Сохраняет stats_df в CSV-файл.
        index=False — не пишем индекс строк в файл.
        """
        if self.stats_df is None:
            self.compute()
        self.stats_df.to_csv(out_path, index=False, encoding='utf-8')
        print(f"[GroupStatsCalculator] group_stats.csv сохранён → {out_path}")


# ══════════════════════════════════════════════════════════════════
#  КЛАСС 4 — MeanDiffAnalyzer  (Неделя 12)
# ══════════════════════════════════════════════════════════════════

class MeanDiffAnalyzer:
    """
    Неделя 12 — Разность средних и фильтрация выбросов.

    ЗАЧЕМ НУЖЕН ЭТОТ КЛАСС:
      Единственное сводное число — «насколько группа A отличается
      от группы B в среднем». Плюс опциональная очистка выбросов
      методом IQR (межквартильного размаха) — стандартный способ
      убрать аномальные наблюдения без субъективного порога.

    МЕТОД IQR:
      Q1 = 25-й процентиль
      Q3 = 75-й процентиль
      IQR = Q3 - Q1
      Выброс: score < Q1 − 1.5·IQR  или  score > Q3 + 1.5·IQR

    АТРИБУТЫ:
      stats_df     (pd.DataFrame) — таблица из GroupStatsCalculator
      raw_df       (pd.DataFrame) — исходные данные
      mean_diff    (float)        — A − B
      filtered_df  (pd.DataFrame) — данные без выбросов
    """

    def __init__(self, stats_df: pd.DataFrame, raw_df: pd.DataFrame):
        self.stats_df    = stats_df
        self.raw_df      = raw_df
        self.mean_diff   = None
        self.filtered_df = None

    def compute_diff(self) -> float:
        """
        Извлекает mean из stats_df, считает A − B.

        :return: float — разность средних
        """
        row_a = self.stats_df[self.stats_df['group'] == 'A']
        row_b = self.stats_df[self.stats_df['group'] == 'B']
        mean_a = float(row_a['mean'].values[0])
        mean_b = float(row_b['mean'].values[0])
        self.mean_diff = round(mean_a - mean_b, 2)

        print("\n" + "═"*54)
        print("  РАЗНОСТЬ СРЕДНИХ  (Неделя 12)")
        print("═"*54)
        print(f"  Среднее группы A : {mean_a}")
        print(f"  Среднее группы B : {mean_b}")
        print(f"  Разность (A − B) : {self.mean_diff}")
        print("═"*54 + "\n")
        return self.mean_diff

    def remove_outliers(self) -> pd.DataFrame:
        """
        Удаляет выбросы по IQR внутри каждой группы.
        Работает отдельно для A и B, чтобы не смешивать распределения.

        :return: отфильтрованный DataFrame
        """
        frames = []
        print("[MeanDiffAnalyzer] Фильтрация выбросов по методу IQR:")
        for group_name, group_df in self.raw_df.groupby('group'):
            scores = group_df['score']
            q1  = scores.quantile(0.25)
            q3  = scores.quantile(0.75)
            iqr = q3 - q1
            lo  = q1 - 1.5 * iqr
            hi  = q3 + 1.5 * iqr
            filtered  = group_df[(scores >= lo) & (scores <= hi)]
            n_removed = len(group_df) - len(filtered)
            print(f"  Группа {group_name}: Q1={q1:.1f}, Q3={q3:.1f}, "
                  f"IQR={iqr:.1f}, границы=[{lo:.1f}, {hi:.1f}], "
                  f"удалено: {n_removed}")
            frames.append(filtered)

        self.filtered_df = pd.concat(frames).reset_index(drop=True)
        print(f"\n  До фильтрации: {len(self.raw_df)} строк")
        print(f"  После:         {len(self.filtered_df)} строк")

        # Пересчёт разности средних на очищенных данных
        new_means = self.filtered_df.groupby('group')['score'].mean().round(2)
        new_diff  = round(new_means['A'] - new_means['B'], 2)
        print(f"  Разность средних (после очистки): {new_diff}\n")

        return self.filtered_df


# ══════════════════════════════════════════════════════════════════
#  КЛАСС 5 — PlotBuilder  (Неделя 13)
# ══════════════════════════════════════════════════════════════════

class PlotBuilder:
    """
    Неделя 13 — Визуализация: гистограммы + boxplot → PNG.

    ЗАЧЕМ НУЖЕН ЭТОТ КЛАСС:
      Визуальное сравнение гораздо нагляднее таблицы чисел.
      - Гистограммы показывают форму распределения.
      - Boxplot показывает медиану, IQR, «усы» и выбросы.

    АТРИБУТЫ:
      scores_a  (np.ndarray) — баллы группы A
      scores_b  (np.ndarray) — баллы группы B
      n_bins    (int)        — число бинов гистограммы
      out_path  (str)        — имя PNG-файла
    """

    def __init__(self, scores_a: np.ndarray, scores_b: np.ndarray,
                 n_bins: int = 10, out_path: str = 'score_plots.png'):
        self.scores_a = scores_a
        self.scores_b = scores_b
        self.n_bins   = n_bins
        self.out_path = out_path

    def _common_bins(self) -> np.ndarray:
        """Общие границы бинов для гистограмм (аналог Неделя 10)."""
        all_scores = np.concatenate([self.scores_a, self.scores_b])
        return np.linspace(np.min(all_scores),
                           np.max(all_scores), self.n_bins + 1)

    def build_and_save(self):
        """
        Строит фигуру 12×5 дюймов с двумя subplot:
          ax1 — гистограммы с alpha=0.6 + вертикальные линии средних
          ax2 — boxplot с раскраской по группам

        Сохраняет в PNG через plt.savefig.
        """
        bin_edges = self._common_bins()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Задание 20: Сравнение баллов групп A и B',
                     fontsize=14, fontweight='bold')

        # ── Левый: гистограммы ──────────────────────────────────
        ax1.hist(self.scores_a, bins=bin_edges,
                 alpha=0.65, color='steelblue', label='Группа A',
                 edgecolor='white', linewidth=0.5)
        ax1.hist(self.scores_b, bins=bin_edges,
                 alpha=0.65, color='coral', label='Группа B',
                 edgecolor='white', linewidth=0.5)

        mean_a = np.mean(self.scores_a)
        mean_b = np.mean(self.scores_b)

        ax1.axvline(mean_a, color='navy', linestyle='--',
                    linewidth=1.5, label=f'mean A = {mean_a:.1f}')
        ax1.axvline(mean_b, color='darkred', linestyle='--',
                    linewidth=1.5, label=f'mean B = {mean_b:.1f}')

        ax1.set_title('Гистограммы score (общая шкала X)')
        ax1.set_xlabel('Балл')
        ax1.set_ylabel('Количество наблюдений')
        ax1.legend(fontsize=8)

        # ── Правый: boxplot ──────────────────────────────────────
        bp = ax2.boxplot(
            [self.scores_a, self.scores_b],
            patch_artist=True,
            tick_labels=['Группа A', 'Группа B'],
            notch=False,
            widths=0.4,
        )
        colors = ['steelblue', 'coral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        ax2.set_title('Boxplot score по группам')
        ax2.set_ylabel('Балл')

        plt.tight_layout()
        plt.savefig(self.out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[PlotBuilder] PNG сохранён → {self.out_path}")


# ══════════════════════════════════════════════════════════════════
#  КЛАСС 6 — JsonReportBuilder  (Неделя 14) ← ГЛАВНЫЙ В ЗАЩИТЕ
# ══════════════════════════════════════════════════════════════════

class JsonReportBuilder:
    """
    Неделя 14 — POST CSV того же формата → JSON-отчёт.

    ЗАЧЕМ НУЖЕН ЭТОТ КЛАСС:
      Имитирует HTTP POST endpoint: принимает данные в CSV-формате
      (в виде строки или файла), парсит их, считает статистику
      и возвращает/сохраняет JSON с:
        - средними по группам (mean_A, mean_B)
        - количеством наблюдений (count_A, count_B)
        - разностью средних (mean_diff = mean_A − mean_B)

    Это финальная «точка сборки» всех предыдущих классов.

    АТРИБУТЫ:
      filepath     (str)  — путь к CSV-файлу (вход)
      report       (dict) — словарь с итоговыми данными
      json_path    (str)  — путь для сохранения JSON-отчёта
    """

    def __init__(self, filepath: str, json_path: str = 'report.json'):
        """
        :param filepath:  входной CSV (scores_groups.csv или аналог)
        :param json_path: куда сохранить JSON
        """
        self.filepath  = filepath
        self.json_path = json_path
        self.report    = {}

    # ------------------------------------------------------------------
    # Приватные вспомогательные методы
    # ------------------------------------------------------------------

    def _parse_csv(self) -> pd.DataFrame:
        """
        Читает CSV через Pandas.
        Проверяет наличие обязательных колонок.

        :return: pd.DataFrame с колонками [group, score]
        :raises: ValueError если колонки отсутствуют
        """
        df = pd.read_csv(self.filepath)
        required = {'group', 'score'}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"В CSV отсутствуют колонки: {missing}")
        print(f"[JsonReportBuilder] CSV прочитан: {len(df)} строк, "
              f"группы: {df['group'].unique().tolist()}")
        return df

    def _compute_group_stats(self, df: pd.DataFrame) -> dict:
        """
        Считает mean, std, count для каждой группы через Pandas.

        :param df: исходный DataFrame
        :return:   словарь {group_name: {mean, std, count}}
        """
        agg = (
            df.groupby('group')['score']
            .agg(['mean', 'std', 'count'])
            .round(4)
        )
        result = {}
        for group_name, row in agg.iterrows():
            result[group_name] = {
                'mean':  float(row['mean']),
                'std':   float(row['std']),
                'count': int(row['count']),
            }
        return result

    # ------------------------------------------------------------------
    # Публичные методы
    # ------------------------------------------------------------------

    def build(self) -> dict:
        """
        Основной метод — собирает JSON-отчёт.

        Алгоритм:
          1. _parse_csv()           → df
          2. _compute_group_stats() → stats
          3. Вычисляем mean_diff = mean_A − mean_B
          4. Собираем итоговый словарь self.report
          5. Выводим в консоль для проверки

        :return: словарь report
        """
        df    = self._parse_csv()
        stats = self._compute_group_stats(df)

        # Разность средних
        mean_a    = stats.get('A', {}).get('mean', None)
        mean_b    = stats.get('B', {}).get('mean', None)
        mean_diff = round(mean_a - mean_b, 4) if (mean_a and mean_b) else None

        # Формируем отчёт — «POST-ответ» в формате JSON
        self.report = {
            'source_file': self.filepath,
            'groups': {},
            'mean_diff_A_minus_B': mean_diff,
        }

        for group_name, group_stats in stats.items():
            self.report['groups'][group_name] = {
                'mean':  group_stats['mean'],
                'std':   group_stats['std'],
                'count': group_stats['count'],
            }

        # Красивый вывод в консоль
        print("\n" + "═"*54)
        print("  JSON ОТЧЁТ  (Неделя 14)")
        print("═"*54)
        for g, s in self.report['groups'].items():
            print(f"  Группа {g}: mean={s['mean']}, "
                  f"std={s['std']}, count={s['count']}")
        print(f"\n  Разность средних (A − B): {self.report['mean_diff_A_minus_B']}")
        print("═"*54 + "\n")

        return self.report

    def save_json(self) -> str:
        """
        Сохраняет self.report в JSON-файл.
        json.dumps с indent=4 — читаемый, форматированный JSON.

        :return: путь к сохранённому JSON
        """
        if not self.report:
            self.build()

        json_str = json.dumps(self.report, ensure_ascii=False, indent=4)
        with open(self.json_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        print(f"[JsonReportBuilder] JSON сохранён → {self.json_path}")
        return self.json_path

    def post_simulate(self, csv_content: str) -> str:
        """
        Дополнительный метод — имитирует HTTP POST:
        принимает содержимое CSV-файла в виде строки,
        парсит через pd.read_csv(StringIO(...)) и возвращает JSON.

        Пример использования:
            csv_str = open('scores_groups.csv').read()
            json_response = builder.post_simulate(csv_str)

        :param csv_content: строка — содержимое CSV
        :return: JSON-строка (как HTTP-ответ сервера)
        """
        from io import StringIO
        df    = pd.read_csv(StringIO(csv_content))
        stats = self._compute_group_stats(df)

        mean_a = stats.get('A', {}).get('mean')
        mean_b = stats.get('B', {}).get('mean')
        diff   = round(mean_a - mean_b, 4) if (mean_a and mean_b) else None

        response = {
            'status': 'ok',
            'groups': stats,
            'mean_diff_A_minus_B': diff,
        }
        json_response = json.dumps(response, ensure_ascii=False, indent=4)
        print("\n[JsonReportBuilder] POST-имитация. Ответ сервера:")
        print(json_response)
        return json_response


# ══════════════════════════════════════════════════════════════════
#  ТОЧКА ВХОДА — запускает ВСЕ недели по порядку
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':


    print("  ЗАДАНИЕ 20: ДВЕ ГРУППЫ ПО ЧИСЛОВОМУ БАЛЛУ")


    CSV_FILE = 'scores_groups.csv'

    # ── Неделя 9: загрузка и базовая статистика ────────────────
    print(" НЕДЕЛЯ 9 — Загрузка CSV, маски NumPy, mean/std")
    loader = ScoreLoader(CSV_FILE)
    week9_stats = loader.compute_stats()

    # ── Неделя 10: гистограммы ─────────────────────────────────
    print(" НЕДЕЛЯ 10 — Общие границы бинов, np.histogram")
    hist_analyzer = HistogramAnalyzer(
        loader.scores_a, loader.scores_b, n_bins=8
    )
    hist_analyzer.compute()
    hist_analyzer.print_histograms()

    # ── Неделя 11: Pandas groupby + сохранение CSV ─────────────
    print(" НЕДЕЛЯ 11 — Pandas groupby/agg → group_stats.csv")
    calc = GroupStatsCalculator(CSV_FILE)
    calc.compute()
    calc.save('group_stats.csv')

    # ── Неделя 12: разность средних + IQR ─────────────────────
    print(" НЕДЕЛЯ 12 — Разность средних и фильтрация выбросов")
    diff_analyzer = MeanDiffAnalyzer(
        stats_df=calc.stats_df,
        raw_df=calc.df
    )
    diff_analyzer.compute_diff()
    diff_analyzer.remove_outliers()

    # ── Неделя 13: графики → PNG ────────────────────────────────
    print(" НЕДЕЛЯ 13 — Matplotlib: гистограммы + boxplot → PNG")
    plotter = PlotBuilder(
        scores_a=loader.scores_a,
        scores_b=loader.scores_b,
        n_bins=8,
        out_path='score_plots.png'
    )
    plotter.build_and_save()

    # ── Неделя 14: JSON отчёт ───────────────────────────────────
    print(" НЕДЕЛЯ 14 — POST CSV → JSON отчёт")
    reporter = JsonReportBuilder(
        filepath=CSV_FILE,
        json_path='report.json'
    )
    reporter.build()
    reporter.save_json()

    # Имитация POST-запроса с содержимым CSV
    print("\n--- Имитация HTTP POST ---")
    csv_content = open(CSV_FILE, 'r', encoding='utf-8').read()
    reporter.post_simulate(csv_content)

    # ── Финальный вывод ────────────────────────────────────────
    print("\n" + "█"*56)
    print("  ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ. ФАЙЛЫ:")
    print("    group_stats.csv — статистика по группам (п.11)")
    print("    score_plots.png — гистограммы + boxplot (п.13)")
    print("    report.json     — JSON отчёт (п.14)")
    print("█"*56 + "\n")