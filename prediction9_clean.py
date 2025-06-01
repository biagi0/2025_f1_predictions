""" Main Imports """
import os
import fastf1
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

""" Sklearn Imports """
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

""" Import custom racepace function """
from racepace import race_pace

""" Plotting Imports """
import matplotlib.pyplot as plt

""" Cache fastf1 """
cache_path = "f1_cache"
os.makedirs(cache_path, exist_ok=True)
fastf1.Cache.enable_cache(cache_path)

""" Dictionary for the races in the season """
races = {
    1: "Australia GP", 2: "Chinese GP", 3: "Japan GP", 4: "Bahrain GP", 5: "Saudi Arabia GP",
    6: "Miami GP", 7: "Emilia Romagna GP", 8: "Monaco GP", 9: "Spanish GP", 10: "Canadian GP",
    11: "Austrian GP", 12: "British GP", 13: "Belgian GP", 14: "Hungary GP", 15: "Dutch GP",
    16: "Italian GP", 17: "Azerbaijan GP", 18: "Singapore GP", 19: "United States GP", 20: "Mexican GP",
    21: "Brazilian GP", 22: "Las Vegas GP", 23: "Qatar GP", 24: "Abu Dhabi GP",
}

""" Dictionary of drivers to teams """
driver_to_team_2025 = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari",
    "RUS": "Mercedes", "HAM": "Ferrari", "GAS": "Alpine", "ALO": "Aston Martin",
    "TSU": "Red Bull", "SAI": "Williams", "HUL": "Kick Sauber", "OCO": "HAAS", 
    "STR": "Aston Martin", "LAW": "Racing Bulls", "HAD": "Racing Bulls", "BEA": "HAAS",
    "BOR": "Kick Sauber", "ANT": "Mercedes", "ALB": "Williams", "COL": "Alpine"
}
driver_to_team_2024 = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari",
    "RUS": "Mercedes", "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin",
    "TSU": "Racing Bulls", "SAI": "Ferrari", "HUL": "Haas", "OCO": "Alpine", 
    "STR": "Aston Martin", "LAW": "Racing Bulls", "MAG": "Haas", "BOT": "Kick Sauber",
    "GUA": "Kick Sauber", "PER": "Red Bull", "ALB": "Williams", "SAR": "Williams"
}

""" Load the 2024 Barcelona GP session & compute sector times """
year = 2024
race = races[9] # From dictionary above
event = "R" # R - Race

session = fastf1.get_session(year, race, event)
session.load()
laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps.dropna(inplace=True)

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps[f"{col} (s)"] = laps[col].dt.total_seconds()

sector_times = laps.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()
sector_times["TotalSectorTime (s)"] = (
    sector_times["Sector1Time (s)"] +
    sector_times["Sector2Time (s)"] +
    sector_times["Sector3Time (s)"]
)

""" Add the clean are race pace from FP2 (racepace.py) """
clean_air_race_pace = race_pace(2025, 9, "FP2", "MEDIUM")

""" Qualifying data for the current race"""
qualifying_2025 = pd.DataFrame({
    "Driver": ["PIA", "NOR", "VER", "RUS", "HAM", "ANT", "LEC",
               "GAS", "HAD", "ALO", "ALB", "BOR", "LAW"],
    "QualifyingTime": [
        71.546, 71.755, 71.848, 71.848, 72.045, 72.111,
        72.131, 72.199, 72.252, 72.284, 72.641, 72.756, 72.763
    ]
})
qualifying_2025["CleanAirRacePace (sec)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

""" Load in the weather forecast for the race """
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
weather_url = (
    f"http://api.openweathermap.org/data/2.5/forecast?lat=41.5700&lon=2.2610"
    f"&appid={API_KEY}&units=metric"
)
response = requests.get(weather_url)
weather_data = response.json()
forecast_time = "2025-06-01 13:00:00" # change to date of the race
forecast_data = next((f for f in weather_data["list"] if f["dt_txt"] == forecast_time), None)

rain_probability = forecast_data["pop"] if forecast_data else 0.0
temperature = forecast_data["main"]["temp"] if forecast_data else 20.0

""" Team performance scoring """
team_points = {
    "McLaren": 319, "Mercedes": 147, "Red Bull": 143, "Williams": 54,
    "Ferrari": 142, "Haas": 26, "Aston Martin": 14, "Kick Sauber": 6,
    "Racing Bulls": 22, "Alpine": 7
}
max_pts = max(team_points.values())
team_perf_score = {team: pts / max_pts for team, pts in team_points.items()}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team_2024)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_perf_score)

""" Merge the 2025 Qualifying with the 2024 Race Data """
merged_data = qualifying_2025.merge(
    sector_times[["Driver", "TotalSectorTime (s)"]],
    on="Driver", how="left"
)
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature
merged_data['QualifyingTime'] = merged_data['QualifyingTime']

""" Only Keep drivers present in 2024 """
valid_drivers = merged_data["Driver"].isin(laps["Driver"].unique())
merged_data = merged_data[valid_drivers] #.reset_index(drop=True)

""" Define our X and Y variables """
X = merged_data[[
    "QualifyingTime", "RainProbability", "Temperature",
    "TeamPerformanceScore", "CleanAirRacePace (sec)"
]].copy()
y = laps.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"]).values

""" Preprocessing pipeline (imputer + scaler) """
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
X_scaled = preprocessor.fit_transform(X)

""" Train Test Split """
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=37
)

""" Gradient Boosting Model """
gbr = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.7, max_depth=3, random_state=37
)
gbr.fit(X_train_scaled, y_train)
y_pred_gbr = gbr.predict(X_test_scaled)
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
print(f"Gradient Boosting Test MAE: {mae_gbr:.3f} seconds")

""" Add GBR predictions back to merged_data (on full X_scaled) """
merged_data["Predicted_GBR (s)"] = gbr.predict(X_scaled)

""" Sort by GBR prediction """
by_gbr = merged_data.sort_values("Predicted_GBR (s)").reset_index(drop=True)
print("\nPredicted 2025 Barcelona GP Winner (GBR)")
print(by_gbr[["Driver", "Predicted_GBR (s)"]].head(3))
print("\nFull list")
print(by_gbr[["Driver", "Predicted_GBR (s)"]])

""" Plot Feature Importance """
plt.figure(figsize=(8,5))
plt.barh(
    ["QualiTime", "RainProb", "Temp", "TeamPerf", "CleanAirPace (sec)"],
    gbr.feature_importances_,
    color="skyblue"
)
plt.xlabel("Importance")
plt.title("GBR Feature Importances")
plt.tight_layout()
plt.show()

""" Plot Effect of Clean Air Pace """
plt.figure(figsize=(12,8))
plt.scatter(
    merged_data["CleanAirRacePace (sec)"],
    merged_data["Predicted_GBR (s)"],
    label="GBR Pred"
)
for i, drv in enumerate(merged_data["Driver"]):
    plt.annotate(
        drv,
        (
            merged_data["CleanAirRacePace (sec)"].iloc[i],
            merged_data["Predicted_GBR (s)"].iloc[i]
        ),
        xytext=(5,5), textcoords="offset points"
    )
plt.xlabel("Clean Air Race Pace (sec)")
plt.ylabel("Predicted Race Time (s) by GBR")
plt.title("Effect of Clean Air Race Pace on GBR Predictions")
plt.tight_layout()
plt.show()