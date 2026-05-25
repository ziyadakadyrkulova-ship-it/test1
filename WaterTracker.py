from WaterEntry import WaterEntry


# WaterTracker класы барлық жазбаларды сақтайды
class WaterTracker:

    # Конструктор
    def __init__(self):

        # Бос тізім
        self.entries = []

    # Жаңа су жазбасын қосу
    def add_entry(self, date, ml, time):

        # WaterEntry объектісін құрамыз
        entry = WaterEntry(date, ml, time)

        # Тізімге қосамыз
        self.entries.append(entry)

    # Барлық жазбаларды шығару
    def show_entries(self):

        for entry in self.entries:
            print(entry.to_dict())

    # Жалпы ішілген суды есептеу
    def total_water(self):

        # Барлық ml мәндерін қосады
        return sum(entry.ml for entry in self.entries)


# Объект құру
tracker = WaterTracker()

# Мәлімет қосу
tracker.add_entry(
    "2026-05-26",
    250,
    "09:00"
)

tracker.add_entry(
    "2026-05-26",
    300,
    "12:30"
)

# Барлық жазбаларды шығару
tracker.show_entries()

# Жалпы су мөлшері
print(
    "Жалпы су:",
    tracker.total_water(),
    "мл"
)