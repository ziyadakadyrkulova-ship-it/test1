import json
from WaterTracker import WaterTracker


# JsonJournal класы JSON файлмен жұмыс істейді
class JsonJournal:

    # Конструктор
    def __init__(self, filename):

        # Файл аты
        self.filename = filename

    # JSON файлға сақтау
    def save_to_json(self, tracker):

        # Барлық объектілерді dict форматына ауыстыру
        data = [
            entry.to_dict()
            for entry in tracker.entries
        ]

        # Файлға жазу
        with open(
            self.filename,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=4
            )

    # JSON файлдан оқу
    def load_from_json(self):

        with open(
            self.filename,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)


# Tracker объектісі
tracker = WaterTracker()

# Мәлімет қосу
tracker.add_entry(
    "2026-05-26",
    200,
    "08:00"
)

tracker.add_entry(
    "2026-05-26",
    300,
    "13:00"
)

# JSON объектісі
journal = JsonJournal("water.json")

# Файлға сақтау
journal.save_to_json(tracker)

# Файлдан оқу
print(journal.load_from_json())