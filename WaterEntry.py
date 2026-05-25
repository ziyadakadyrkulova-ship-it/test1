from datetime import datetime
# WaterEntry класы бір су ішу жазбасын сақтайды
class WaterEntry:

    # Конструктор
    def __init__(self, date: str, ml: int, time: str):

        self.date = self.validate_date(date)  # Күнді тексеру
        self.ml = self.validate_ml(ml) # Су мөлшерін тексеру
        self.time = self.validate_time(time) # Уақытты тексеру

    # Күн форматын тексеру функциясы
    def validate_date(self, date: str):

        try:
            # YYYY-MM-DD форматында ма тексереді
            datetime.strptime(date, "%Y-%m-%d")
            return date

        except ValueError:
            raise ValueError(
                "Күн форматы қате. Мысалы: 2026-05-26"
            )

    # Су мөлшерін тексеру
    def validate_ml(self, ml: int):

        # Егер 0 немесе теріс болса
        if ml <= 0:
            raise ValueError(
                "Су мөлшері 0-ден үлкен болуы керек"
            )

        return ml

    # Уақыт форматын тексеру
    def validate_time(self, time: str):

        try:
            # HH:MM форматында ма тексереді
            datetime.strptime(time, "%H:%M")
            return time

        except ValueError:
            raise ValueError(
                "Уақыт форматы қате. Мысалы: 10:30"
            )

    # Объектіні dict форматына айналдыру
    def to_dict(self):

        return {
            "date": self.date,
            "ml": self.ml,
            "time": self.time
        }


# Объект құру
entry = WaterEntry(
    "2026-05-26",
    250,
    "10:30"
)
# Нәтиже шығару
print(entry.to_dict())


