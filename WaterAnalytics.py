import pandas as pd
# DailyStats класы статистика есептейді
class DailyStats:

    # Конструктор
    def __init__(self, data):

        # DataFrame жасау
        self.df = pd.DataFrame(data)

    # Күндік сумманы есептеу
    def daily_sum(self):

        # date бойынша топтау
        result = self.df.groupby("date")["ml"].sum()

        return result

    # Орташа су мөлшерін есептеу
    def average_water(self):

        return self.df["ml"].mean()

# Мәліметтер
data = [
    {
        "date": "2026-05-26",
        "ml": 250,
        "time": "09:00"
    },

    {
        "date": "2026-05-26",
        "ml": 300,
        "time": "12:00"
    },

    {
        "date": "2026-05-27",
        "ml": 400,
        "time": "10:00"
    }
]

# Объект
stats = DailyStats(data)

# Күндік сумма
print(stats.daily_sum())

# Орташа су
print(
    "Орташа:",
    stats.average_water(),
    "мл"
)