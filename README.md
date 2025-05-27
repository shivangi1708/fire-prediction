# 🔥 Fire Prediction Web Application

This is a Flask-based web application that predicts whether a fire will occur based on various environmental and forest-related input features. The machine learning model behind the app outputs `1` for **Rain (fire less likely)** and `0` for **No Rain (fire more likely)**.

---

## 🚀 Features

- Interactive HTML form for environmental data input
- Backend Flask app using a trained ML model
- Real-time prediction results: "Rain" or "No Rain"
- Clean and responsive interface with CSS styling

---

## 🧾 Input Parameters

| Parameter         | Type   | Description                              |
|------------------|--------|------------------------------------------|
| `collector_id`     | float  | ID of the data collector                  |
| `month`           | float  | Month number (e.g., 1 = January)          |
| `time_of_day`     | str    | Time of day: `morning`, `afternoon`, `evening`, `night` |
| `temperature`     | float  | Temperature in degrees                    |
| `humidity`        | float  | Humidity percentage                       |
| `wind_intensity`  | float  | Wind speed/intensity                      |
| `rain`            | float  | Recent rainfall measurement               |
| `surface_litter`  | float  | Forest surface litter amount              |
| `tree_age`        | float  | Average tree age in the area              |
| `tree_density`    | float  | Number of trees per area unit             |
| `l_score`         | float  | Litter score                              |
| `c_score`         | float  | Canopy score                              |

---

## 🖥️ How It Works

1. User fills in the form on the homepage.
2. Data is sent to `/predictdata` via POST.
3. Flask app processes input, converts data, and calls the trained model.
4. Result (`1` or `0`) is shown as "Rain" or "No Rain".

---

💡 Example Output
After submitting values:

🔍 Prediction: Rain
OR
🔍 Prediction: No Rain
