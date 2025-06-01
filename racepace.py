import fastf1
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

def race_pace(year, race, event, compound):
    # Load session
    session = fastf1.get_session(year, race, event)
    session.load()

    # Filter accurate laps, exclude pit-ins/outs, and missing compound
    laps = session.laps.pick_accurate()
    laps = laps[laps['PitInTime'].isna() & laps['PitOutTime'].isna()]
    laps = laps.dropna(subset=['Compound'])

    # Keep only target compound
    laps = laps[laps['Compound'].str.upper() == compound.upper()]

    # Convert lap times to seconds
    laps['LapTime (s)'] = laps['LapTime'].dt.total_seconds()

    # Keep drivers with at least 3 laps
    drivers = [d for d in laps['Driver'].unique() if len(laps[laps['Driver'] == d]) >= 3]

    # Compute mean lap time per driver
    avg_pace = laps[laps['Driver'].isin(drivers)].groupby('Driver')['LapTime (s)'].mean()
    
    # Round to 3 decimals for readability
    clean_air_race_pace = {
        drv: round(time, 3) for drv, time in avg_pace.sort_values().items()
    }

    return clean_air_race_pace