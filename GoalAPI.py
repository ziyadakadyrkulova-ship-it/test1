from flask import Flask, send_file
import matplotlib # matplotlib кітапханасы
matplotlib.use("Agg")# Flask ішінде график дұрыс шығу үшін
import matplotlib.pyplot as plt # График салу кітапханасы
app = Flask(__name__)# Flask объектісі

class WaterTracker:
    # Су ішу мәліметтері
    def __init__(self):
        self.data = [
            {"time": "08:00", "ml": 250},
            {"time": "11:00", "ml": 300},
            {"time": "14:00", "ml": 450},
            {"time": "18:00", "ml": 500}
        ]

    # PNG график жасау функциясы
    def create_plot(self):
        times = [item["time"] for item in self.data] # Уақыттарды алу
        values = [item["ml"] for item in self.data] # Су мөлшерлерін алу

        plt.figure(figsize=(8, 5))  # Figure өлшемі
        plt.plot(times, values, marker="o")
        plt.title("Water Consumption Line")
        plt.xlabel("Time")
        plt.ylabel("ML")
        plt.grid(True) # Тор сызықтар

        plt.savefig("water.png")# PNG файлға сақтау
        plt.close() # Жадыны тазарту

# Объект құру
tracker = WaterTracker()
# Басты бет route
@app.route("/")
def home():
    return "Water Tracker API"

# PNG график route
@app.route("/plot")
def plot():
    tracker.create_plot()  # График жасау
    return send_file("water.png", mimetype="image/png")# PNG файлды браузерге жіберу

# Программаны іске қосу
if __name__ == "__main__":
    app.run(debug=True)