import os
import sys
import requests
import pandas as pd
import numpy as np
import joblib

from dotenv import load_dotenv

load_dotenv()
#----loading the ml model for rain prediction
model_path="backend/xgb_model (2).joblib"
if os.path.exists(model_path):
    model=joblib.load(model_path)
    print("imported")
else:
    model=None
    print("Not imported")

GIS_ALERTS_URL = os.environ.get("GIS_ALERTS_URL", "https://example.com/gis/alerts")

def fetch_gis_alert_data():
    """
    Helper used by tests. Calls requests.get so tests can patch requests.get.
    Returns (data, status_code).

    Behavior expected by tests:
      - requests.exceptions.ConnectionError -> (None, 503)
      - requests.exceptions.Timeout         -> (None, 504)
      - On success: (response.json() or response.text, response.status_code)
      - On other exceptions: (None, 500)
    """
    try:
        resp = requests.get(GIS_ALERTS_URL, timeout=10)
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError:
            data = resp.text
        return data, resp.status_code

    except requests.exceptions.ConnectionError:
        return None, 503

    except requests.exceptions.Timeout:
        return None, 504

    except Exception:
        return None, 500
    
from flask import (
    Flask,
    jsonify,
    request,
    send_from_directory
)

from flask_cors import CORS

# =========================================================
# APP CONFIG
# =========================================================

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "Frontend"
)

CHATBOT_DIR = os.path.join(BASE_DIR, "AI-chatbot")
if CHATBOT_DIR not in sys.path:
    sys.path.insert(0, CHATBOT_DIR)

from chatbot import handle_chatbot_request

# =========================================================
# FRONTEND ROUTES
# =========================================================

@app.route("/")
def home():
    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


@app.route("/Analysis/<path:filename>")
def analysis_files(filename):
    return send_from_directory(
        os.path.join(FRONTEND_DIR, "Analysis"),
        filename
    )


@app.route("/<path:filename>")
def frontend_files(filename):
    return send_from_directory(
        FRONTEND_DIR,
        filename
    )

# =========================================================
# THRESHOLDS
# =========================================================

FLOOD_RISK_THRESHOLD     = 0.65
HEAT_RISK_THRESHOLD      = 0.75
WILDFIRE_RISK_THRESHOLD  = 0.65
CYCLONE_RISK_THRESHOLD   = 0.60
DROUGHT_RISK_THRESHOLD   = 0.70

# =========================================================
# GET LOCATION COORDINATES (Nominatim / OpenStreetMap)
# =========================================================

def get_coordinates(city, state, country):

    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "ClimateWeatherApp/1.0"}

    # Helper function to make the request
    def fetch_coords(params):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
            data = resp.json()
            if data and len(data) > 0:
                location = data[0]
                return {
                    "latitude":  float(location["lat"]),
                    "longitude": float(location["lon"]),
                    "city":      location.get("display_name", city).split(",")[0].strip(),
                    "state":     state,
                    "country":   country
                }
        except Exception as e:
            print(f"Geocoding request failed: {e}")
        return None

    # Attempt 1: Free text search (most robust for Nominatim)
    query_parts = [city]
    if state: query_parts.append(state)
    if country: query_parts.append(country)
    query = ", ".join(query_parts)
    
    print(f"Geocoding attempt 1: q='{query}'")
    loc_data = fetch_coords({"q": query, "format": "json", "limit": 1})
    if loc_data:
        print(f"Successfully geocoded: {loc_data['city']} -> ({loc_data['latitude']}, {loc_data['longitude']})")
        return loc_data

    # Attempt 2: Structured search (city)
    if state:
        print("Attempt 1 failed. Retrying with structured 'city' search...")
        loc_data = fetch_coords({"city": city, "state": state, "country": country, "format": "json", "limit": 1})
        if loc_data:
            return loc_data

    # Attempt 3: Structured search (town)
    if state:
        print("Attempt 2 failed. Retrying with structured 'town' search...")
        loc_data = fetch_coords({"town": city, "state": state, "country": country, "format": "json", "limit": 1})
        if loc_data:
            return loc_data

    # Attempt 4: Fallback (if no state, or if all above failed and we want to try ignoring state)
    if not state:
        fallback_query = f"{city}, {country}"
        print(f"Retrying with fallback query: {fallback_query}")
        loc_data = fetch_coords({"q": fallback_query, "format": "json", "limit": 1})
        if loc_data:
            return loc_data

    print("All geocoding attempts failed. Location not found.")
    return None

# =========================================================
# GIS ALERT DATA (Issue #83: Exception Handling)
# =========================================================

def fetch_gis_alert_data():
    """
    Fetches external GIS climate data streams.
    Implements try-except blocks to prevent backend crashes.
    """
    GIS_API_URL = "https://external-gis-source.com"

    try:
        response = requests.get(GIS_API_URL, timeout=5)
        response.raise_for_status()
        return response.json(), 200

    except requests.exceptions.Timeout:
        return {"error": "External GIS service timed out. Please try again."}, 504

    except (requests.exceptions.RequestException, ValueError):
        return {"error": "External GIS service is unavailable or returned an invalid response."}, 503

# =========================================================
# WEATHER API
# =========================================================

@app.route("/weather", methods=["POST"])
def get_weather_insights():

    try:

        payload = request.get_json() or {}

        city = payload.get("city", "").strip()
        state = payload.get("state", "").strip()
        country = payload.get("country", "").strip()

        if not city or not state or not country:

            return jsonify({
                "success": False,
                "message": "Please fill all fields."
            }), 400

        api_key = os.environ.get("OPENWEATHER_API_KEY")

        if not api_key:
            print("OPENWEATHER_API_KEY missing")

            return jsonify({
                "success": False,
                "message": "Weather service configuration error."
            }), 500

# ----------------------------------------------------
# STEP 1: Convert city → coordinates
# ----------------------------------------------------

        geo_response = requests.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={
                "q": f"{city},{state},{country}",
                "limit": 1,
                "appid": api_key
            },
            timeout=15
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()

        if not geo_data:
            return jsonify({
                "success": False,
                "message": "Location not found."
            }), 404

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]

        # ----------------------------------------------------
        # STEP 2: Current weather
        # ----------------------------------------------------

        weather_response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "lat": lat,
                "lon": lon,
                "units": "metric",
                "appid": api_key
            },
            timeout=15
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()

        temp_val = weather_data["main"]["temp"]
        humid_val = weather_data["main"]["humidity"]

        wind_val = round(
            weather_data["wind"]["speed"] * 3.6,
            1
        )

        rain_val = (
            weather_data.get("rain", {}).get("1h")
            or weather_data.get("rain", {}).get("3h")
            or 0
        )

        # ----------------------------------------------------
        # STEP 3: Forecast
        # ----------------------------------------------------

        forecast_response = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={
                "lat": lat,
                "lon": lon,
                "units": "metric",
                "appid": api_key
            },
            timeout=15
        )

        forecast_response.raise_for_status()

        forecast_data = forecast_response.json()

        # ----------------------------------------------------
        # RISK CALCULATIONS
        # ----------------------------------------------------

        flood_risk_metric = round(
            min(
                1.0,
                (
                    rain_val * 0.6 +
                    humid_val * 0.3 +
                    wind_val * 0.1
                ) / 100
            ),
            3
        )

        heat_risk_metric = round(
            min(
                1.0,
                (
                    max(temp_val - 25, 0) * 2 +
                    humid_val * 0.3
                ) / 100
            ),
            3
        )

        wildfire_risk_metric = round(
            min(
                1.0,
                (
                    max(temp_val - 32, 0) * 1.5 +
                    (100 - humid_val) * 0.5 +
                    wind_val * 0.2
                ) / 100
            ),
            3
        )

        cyclone_risk_metric = round(
            min(
                1.0,
                (
                    wind_val * 1.5 +
                    rain_val * 0.5
                ) / 100
            ),
            3
        )

        drought_risk_metric = round(
            min(
                1.0,
                (
                    max(temp_val - 28, 0) +
                    (100 - humid_val)
                ) / 100
            ),
            3
        )

        # ----------------------------------------------------
        # ALERTS
        # ----------------------------------------------------

        calculated_alerts = []

        if flood_risk_metric >= 0.6:
            calculated_alerts.append(
                "⚠ High Flood Risk Detected"
            )

        if heat_risk_metric >= 0.6:
            calculated_alerts.append(
                "🔥 Heatwave Conditions Possible"
            )

        if wildfire_risk_metric >= 0.6:
            calculated_alerts.append(
                "🌲 Elevated Wildfire Risk"
            )

        if cyclone_risk_metric >= 0.6:
            calculated_alerts.append(
                "🌀 Cyclone Risk Detected"
            )

        if drought_risk_metric >= 0.6:
            calculated_alerts.append(
                "☀ Drought Conditions Possible"
            )

        if not calculated_alerts:
            calculated_alerts.append(
                "✅ No major climate threats detected."
            )

        # ----------------------------------------------------
        # FORECAST GENERATION
        # ----------------------------------------------------

        forecast = []

        forecast_items = forecast_data.get("list", [])

        for item in forecast_items[::8][:5]:

            day_temp = item["main"]["temp"]
            day_humidity = item["main"]["humidity"]

            day_rain = (
                item.get("rain", {})
                .get("3h", 0)
            )

            day_wind = round(
                item["wind"]["speed"] * 3.6,
                1
            )

            forecast.append({
                "date": item["dt_txt"],
                "temperature": round(day_temp, 1),
                "humidity": day_humidity,
                "rainfall": round(day_rain, 1),
                "wind_speed": day_wind,
                "risks": {
                    "flood_risk": round(
                        min(
                            1.0,
                            (
                                day_rain * 0.6 +
                                day_humidity * 0.3 +
                                day_wind * 0.1
                            ) / 100
                        ),
                        3
                    ),
                    "heat_risk": round(
                        min(
                            1.0,
                            (
                                max(day_temp - 25, 0) * 2 +
                                day_humidity * 0.3
                            ) / 100
                        ),
                        3
                    ),
                    "wildfire_risk": round(
                        min(
                            1.0,
                            (
                                max(day_temp - 32, 0) * 1.5 +
                                (100 - day_humidity) * 0.5 +
                                day_wind * 0.2
                            ) / 100
                        ),
                        3
                    ),
                    "cyclone_risk": round(
                        min(
                            1.0,
                            (
                                day_wind * 1.5 +
                                day_rain * 0.5
                            ) / 100
                        ),
                        3
                    ),
                    "drought_risk": round(
                        min(
                            1.0,
                            (
                                max(day_temp - 28, 0) +
                                (100 - day_humidity)
                            ) / 100
                        ),
                        3
                    )
                }
            })

        return jsonify({

            "success": True,

            "location": {
                "city": geo_data[0].get("name", city),
                "state": geo_data[0].get("state", state),
                "country": geo_data[0].get("country", country),
                "latitude": lat,
                "longitude": lon
            },

            "weather": {
                "temperature": temp_val,
                "humidity": humid_val,
                "rainfall": rain_val,
                "wind_speed": wind_val
            },

            "risks": {
    "flood_risk": round(flood_risk_metric, 3),
    "flood_risk_confidence": round(flood_risk_metric * 100, 1),
    "flood_risk_level": "HIGH" if flood_risk_metric >= 0.6 else "MEDIUM" if flood_risk_metric >= 0.3 else "LOW",
    "heat_risk": round(heat_risk_metric, 3),
    "heat_risk_confidence": round(heat_risk_metric * 100, 1),
    "heat_risk_level": "HIGH" if heat_risk_metric >= 0.6 else "MEDIUM" if heat_risk_metric >= 0.3 else "LOW",
    "wildfire_risk": round(wildfire_risk_metric, 3),
    "wildfire_risk_confidence": round(wildfire_risk_metric * 100, 1),
    "wildfire_risk_level": "HIGH" if wildfire_risk_metric >= 0.6 else "MEDIUM" if wildfire_risk_metric >= 0.3 else "LOW",
    "cyclone_risk": round(cyclone_risk_metric, 3),
    "cyclone_risk_confidence": round(cyclone_risk_metric * 100, 1),
    "cyclone_risk_level": "HIGH" if cyclone_risk_metric >= 0.6 else "MEDIUM" if cyclone_risk_metric >= 0.3 else "LOW",
    "drought_risk": round(drought_risk_metric, 3),
    "drought_risk_confidence": round(drought_risk_metric * 100, 1),
"drought_risk_level": "HIGH" if drought_risk_metric >= 0.6 else "MEDIUM" if drought_risk_metric >= 0.3 else "LOW",
            },

            "forecast": forecast,

            "alerts": calculated_alerts,
        })

    except Exception as general_err:
        print("Weather Route Error:")
        print(str(general_err))
        return jsonify({"success": False, "message": "Internal server error."}), 500

@app.route("/reverse-geocode", methods=["POST"])
def reverse_geocode():

    try:

        data = request.get_json()

        latitude = data.get("latitude")
        longitude = data.get("longitude")

        if latitude is None or longitude is None:

            return jsonify({
                "success": False,
                "message":
                "Latitude and longitude are required."
            })

        api_key = os.environ.get(
            "OPENWEATHER_API_KEY"
        )

        response = requests.get(
            "https://api.openweathermap.org/geo/1.0/reverse",
            params={
                "lat": latitude,
                "lon": longitude,
                "limit": 1,
                "appid": api_key
            },
            timeout=20
        )

        if response.status_code != 200:

            return jsonify({
                "success": False,
                "message":
                "Reverse geocoding failed."
            })

        result = response.json()

        if not result:

            return jsonify({
                "success": False,
                "message":
                "Location not found."
            })

        location = result[0]

        return jsonify({

            "success": True,

            "city":
            location.get("name", ""),

            "state":
            location.get("state", ""),

            "country":
            location.get("country", "")

        })

    except Exception:

        return jsonify({
            "success": False,
            "message":
            "Reverse geocoding failed."
        })

@app.route("/city-suggestions", methods=["GET"])
def city_suggestions():

    query = request.args.get("q", "").strip()

    if len(query) < 2:
        return jsonify([])

    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": query,
                "format": "json",
                "addressdetails": 1,
                "limit": 5,
                "countrycodes": "in"
            },
            headers={
                "User-Agent": "ClimateShield/1.0"
            },
            timeout=10
        )

        data = response.json()

        suggestions = []

        for item in data:
            address = item.get("address", {})

            if not (
                address.get("city")
                or address.get("town")
                or address.get("village")
                or address.get("municipality")
            ):
                continue

            city_name = (
                address.get("city")
                or address.get("town")
                or address.get("village")
                or address.get("municipality")
            )

            suggestions.append({
                "city": city_name,
                "state": address.get("state", ""),
                "country": address.get("country", "")
            })

        suggestions.sort(
            key=lambda x: (
                not x["city"].lower().startswith(query.lower()),
                x["city"].lower()
            )
        )

        print("Query:", query)
        print("Suggestions:", suggestions)

        return jsonify(suggestions)

    except Exception as e:
        print("City Suggestions Error:", e)
        return jsonify([])
    
# =========================================================
# CHATBOT API
# =========================================================

@app.route("/chatbot", methods=["POST"])
def chatbot():

    try:
        data = request.get_json(silent=True) or {}
        payload, status = handle_chatbot_request(data)
        return jsonify(payload), status

    except Exception:

        return jsonify({
            "success": False,
            "message":
            "Chatbot unavailable."
        })


#----------ML Rain Predctor Feature------#


def extract_features(location_dict):# extract the features from open weather
  loc_data=get_coordinates(location_dict)

  lat=loc_data.get("lat","")
  lon=loc_data.get("lon","")
  if not lat or not lon:
    return "Coordinates not fetched."
  API_KEY = os.getenv("OPENWEATHER_API_KEY")
  url=f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
  response=requests.get(url).json()
  if response.get('cod') != 200:
    return "Failed api call"
  openweather_temp=response['main']['temp']
  openweather_humidity = response['main']['humidity']
  openweather_wind_speed = response['wind']['speed']*3.6
  openweather_wind_dir = response['wind']['deg']
  openweather_gusts = response['wind'].get('gust', openweather_wind_speed)*3.6
  lat_rad = np.radians(lat)
  lon_rad = np.radians(lon)
  coord_x= np.cos(lat_rad) * np.cos(lon_rad)
  coord_y= np.cos(lat_rad) * np.sin(lon_rad)
  coord_z= np.sin(lat_rad)

  live_features = pd.DataFrame({
"latitude":lat,
"longitude":lon,
"wind_direction_10m_dominant": openweather_wind_dir,
"wind_gusts_10m_mean": openweather_gusts,
"wind_speed_10m_mean": openweather_wind_speed,
"temperature_2m_mean": openweather_temp,
"relative_humidity_2m_mean": openweather_humidity,
"coord_x":coord_x,
"coord_y":coord_y,
"coord_z":coord_z}, index=[0])
  result=model.predict(live_features)
  return result[0]
  
def get_rain_criteria(location_dict):
  '''Takes the result from the extract_feature function and predict the likeability of rain.'''
  rain_mm=extract_coordinates(location_dict)
  try:
   if rain_mm <= 2.4:
      return "No Rain / Light Drizzle ☀️"
   elif 2.4 < rain_mm <= 15.5:
        return "Light Rain 🌧️"
   elif 15.5 < rain_mm <= 64.4:
        return "Moderate Rain ⛈️"
   elif 64.4 < rain_mm <= 115.5:
        return "Heavy Rain Alert 🚨"
   elif 115.5 < rain_mm <= 204.4:
        return "Very Heavy Rain Warning 🌊"
   else:
        return "Extremely Heavy Rain / Flood Risk ⚠️"
  except Exception as e:
    return "An unexpected error occurred while determining rain status."







# =========================================================
# LOCAL RUN
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True

    )
