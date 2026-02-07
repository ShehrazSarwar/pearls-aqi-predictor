import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import openmeteo_requests
import requests_cache
from retry_requests import retry
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# 1. SETUP
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["aqi_predictor"]
raw_collection = db["raw_data"]

# API Client Setup
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


def fetch_raw_data(days=730):
    lat, lon = 24.8607, 67.0011
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "timezone": "auto"
    }

    # Fetch responses
    w_resp = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params={**params,
                                                                                            "hourly": ["temperature_2m",
                                                                                                       "relative_humidity_2m",
                                                                                                       "windspeed_10m",
                                                                                                       "winddirection_10m"]})[
        0]
    a_resp = openmeteo.weather_api("https://air-quality-api.open-meteo.com/v1/air-quality", params={**params,
                                                                                                    "hourly": ["pm2_5",
                                                                                                               "pm10",
                                                                                                               "carbon_monoxide",
                                                                                                               "nitrogen_dioxide",
                                                                                                               "sulphur_dioxide",
                                                                                                               "ozone",
                                                                                                               "dust"]})[
        0]

    # Get hourly data blocks
    h_w = w_resp.Hourly()

    # 1. Create the datetime range as native pandas objects
    # This allows you to keep 'datetime' as a proper timestamp in the df
    time_range = pd.date_range(
        start=pd.to_datetime(h_w.Time(), unit="s", utc=True),
        end=pd.to_datetime(h_w.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=h_w.Interval()),
        inclusive="left"
    )

    # 2. Store it as it is in the df (as native timestamps)
    df = pd.DataFrame({
        "datetime": time_range,
        "temperature": h_w.Variables(0).ValuesAsNumpy(),
        "humidity": h_w.Variables(1).ValuesAsNumpy(),
        "wind_speed": h_w.Variables(2).ValuesAsNumpy(),
        "wind_dir": h_w.Variables(3).ValuesAsNumpy(),
        "pm2_5": a_resp.Hourly().Variables(0).ValuesAsNumpy(),
        "pm10": a_resp.Hourly().Variables(1).ValuesAsNumpy(),
        "co": a_resp.Hourly().Variables(2).ValuesAsNumpy(),
        "no2": a_resp.Hourly().Variables(3).ValuesAsNumpy(),
        "so2": a_resp.Hourly().Variables(4).ValuesAsNumpy(),
        "o3": a_resp.Hourly().Variables(5).ValuesAsNumpy(),
        "dust": a_resp.Hourly().Variables(6).ValuesAsNumpy()
    })

    return df


from datetime import timezone

if __name__ == "__main__":
    # Check if we already have data
    count = raw_collection.count_documents({})

    # If DB is empty, fetch 2 years (730 days)
    # If DB has data, just fetch last 3 days to get the newest hour
    days_to_fetch = 730 if count == 0 else 3

    print(f"Database has {count} records. Fetching last {days_to_fetch} days...")
    raw_df = fetch_raw_data(days=days_to_fetch)

    df_for_mongo = raw_df.copy()

    # 1. Fetch existing times and force them to be UTC ISO strings
    print("Standardizing database timestamps...")
    existing_times = set()
    for dt in raw_collection.distinct("datetime"):
        # Force every DB date to UTC and standard string format
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        existing_times.add(dt.astimezone(timezone.utc).isoformat())

    # 2. Filter new records using the exact same standard
    records_to_insert = []
    for record in df_for_mongo.to_dict("records"):
        dt_obj = record["datetime"]
        # Ensure new data is also UTC
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)

        current_time_iso = dt_obj.astimezone(timezone.utc).isoformat()

        if current_time_iso not in existing_times:
            # Important: Keep the datetime object for MongoDB insertion
            record["datetime"] = dt_obj.astimezone(timezone.utc)
            records_to_insert.append(record)

    # 3. Insert only the truly new records
    if records_to_insert:
        raw_collection.insert_many(records_to_insert)
        print(f"Success! Added {len(records_to_insert)} NEW records.")
    else:
        print("Everything is up to date. 0 records added.")

    # Check the most recent record in your DB
    last_record = raw_collection.find_one(sort=[("datetime", -1)])
    print(f"The most recent data in the DB is from: {last_record['datetime']}")