from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import joblib
import numpy as np
from rl_agent import rl_recommendation
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AQICN_TOKEN = "66f7da3b6af1026b572b9e24f32e8726d4154e4b"

model = joblib.load("risk_model.pkl")


# ---------------------------------------------------
# Common Animals (Fauna) - Scientific Names
# ---------------------------------------------------
COMMON_ANIMALS = [
    # Big Cats
    "Panthera leo", "Panthera tigris", "Panthera pardus", "Acinonyx jubatus", "Puma concolor",
    # Elephants
    "Elephas maximus", "Loxodonta africana",
    # Primates
    "Pan troglodytes", "Gorilla gorilla", "Pongo pygmaeus", "Papio hamadryas", "Macaca mulatta",
    # Bears
    "Ursus maritimus", "Ursus arctos", "Ailuropoda melanoleuca", "Melursus ursinus",
    # Ungulates
    "Equus quagga", "Giraffa camelopardalis", "Cervus elaphus", "Antilope cervicapra", "Rhinoceros unicornis",
    # Canines
    "Canis lupus", "Vulpes vulpes", "Cuon alpinus", "Canis aureus",
    # Birds
    "Pavo cristatus", "Aquila chrysaetos", "Psittacus erithacus", "Phoenicopterus roseus", "Grus grus",
    # Reptiles
    "Crocodylus niloticus", "Testudo graeca", "Python molurus", "Naja naja",
    # Other Mammals
    "Hippopotamus amphibius", "Camelus dromedarius", "Bison bison", "Bos mutus", "Lutra lutra",
    # Amphibians
    "Rana temporaria", "Bufo bufo", "Salamandra salamandra",
    # Small Animals
    "Sciurus vulgaris", "Oryctolagus cuniculus", "Erinaceus europaeus", "Hystrix cristata",
]


# ---------------------------------------------------
# Fetch Species
# ---------------------------------------------------
def fetch_species(category: str, limit: int = 30):

    if category == "flora":
        kingdomKey = 6
    else:
        kingdomKey = 1

    url = (
        f"https://api.gbif.org/v1/occurrence/search?"
        f"country=IN&kingdomKey={kingdomKey}&limit={limit}"
    )

    species_set = set(COMMON_ANIMALS) if category == "fauna" else set()

    try:
        data = requests.get(url).json()

        for rec in data.get("results", []):
            sp = rec.get("species") or rec.get("scientificName")
            if sp:
                species_set.add(sp)
    except Exception as e:
        print(f"Error fetching from GBIF: {e}")

    return sorted(list(species_set))


# ---------------------------------------------------
# Fetch States
# ---------------------------------------------------
def fetch_states(limit: int = 50):

    url = f"https://api.gbif.org/v1/occurrence/search?country=IN&limit={limit}"

    data = requests.get(url).json()

    state_set = set()

    for rec in data.get("results", []):
        state = rec.get("stateProvince")
        if state:
            state_set.add(state)

    return sorted(list(state_set))


# ---------------------------------------------------
# Options API
# ---------------------------------------------------
@app.get("/options")
def get_options(category: str = Query(...)):

    return {
        "species": fetch_species(category),
        "states": fetch_states()
    }


# ---------------------------------------------------
# Weather API
# ---------------------------------------------------
def get_weather(lat, lon):

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

    data = requests.get(url).json()

    return data["current_weather"]["temperature"]


# ---------------------------------------------------
# AQI API
# ---------------------------------------------------
def get_aqi(lat, lon):

    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={AQICN_TOKEN}"

    data = requests.get(url).json()

    if data["status"] != "ok":
        return None

    return data["data"]["aqi"]


# ---------------------------------------------------
# Threat Status
# ---------------------------------------------------
def get_threat_status(species: str, aqi: int = None):
    """
    Get threat status from Wikidata with fallback to environment-based estimation.
    """
    score_map = {
        "least concern": 1,
        "near threatened": 2,
        "vulnerable": 3,
        "endangered": 4,
        "critically endangered": 5,
    }

    # Try Wikidata queries with multiple language variations
    queries = [
        f"""
        SELECT ?statusLabel WHERE {{
          ?sp wdt:P225 "{species}".
          ?sp wdt:P141 ?status.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 1
        """,
        f"""
        SELECT DISTINCT ?statusLabel WHERE {{
          ?sp rdfs:label "{species}"@en.
          ?sp wdt:P141 ?status.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 1
        """
    ]

    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/json"}

    for query in queries:
        try:
            r = requests.get(url, params={"query": query}, headers=headers, timeout=5)
            if r.status_code == 200:
                bindings = r.json().get("results", {}).get("bindings", [])
                if bindings:
                    status = bindings[0]["statusLabel"]["value"]
                    return status, score_map.get(status.lower(), 2)
        except:
            continue

    # Fallback: Estimate threat based on environmental indicators
    if aqi is not None:
        if aqi > 200:
            return "High Pollution Risk", 4
        elif aqi > 150:
            return "Moderate Threat", 3
        elif aqi > 100:
            return "Low Threat", 2
    
    return "Not Available", 2


# ---------------------------------------------------
# Analyze API (MULTIPLE LOCATIONS)
# ---------------------------------------------------
@app.get("/analyze")
def analyze(species: str, state: str, category: str):

    if state == "all":

        url = (
            f"https://api.gbif.org/v1/occurrence/search?"
            f"country=IN&scientificName={species}&limit=20"
        )

    else:

        url = (
            f"https://api.gbif.org/v1/occurrence/search?"
            f"country=IN&scientificName={species}&stateProvince={state}&limit=20"
        )

    data = requests.get(url).json()

    results = []

    for rec in data.get("results", []):

        lat = rec.get("decimalLatitude")
        lon = rec.get("decimalLongitude")

        if not lat or not lon:
            continue

        state_name = rec.get("stateProvince", "Unknown")

        temp = get_weather(lat, lon)
        aqi = get_aqi(lat, lon)

        # Get threat status with environment-based estimation
        threat_status, threat_score = get_threat_status(species, aqi)

        cat_val = 0 if category == "flora" else 1

        # Add environmental weighting for more variation
        env_multiplier = 1.0
        if aqi and aqi > 150:
            env_multiplier = 1.3  # Increase risk in high pollution areas
        if temp and temp > 35:
            env_multiplier *= 1.2  # Increase risk in very hot areas

        features = np.array([[temp, aqi, cat_val, threat_score]])

        pred = model.predict(features)[0]
        
        # Apply environmental weighting to vary predictions
        pred_score = pred + (env_multiplier - 1.0) * 0.5
        pred = int(min(2, pred_score))  # Cap at 2 (HIGH)

        risk_map = {
            0: "LOW",
            1: "MEDIUM",
            2: "HIGH"
        }

        risk = risk_map[pred]

        # Context-aware RL action with threat status consideration
        if threat_score >= 4:  # Vulnerable or Endangered
            if risk == "HIGH":
                action = "Urgent Conservation Intervention"
            elif risk == "MEDIUM":
                action = "Deploy Anti-Poaching Patrols"
            else:
                action = "Increase Habitat Protection"
        else:
            action = rl_recommendation(risk)

        results.append({
            "species": species,
            "state": state_name,
            "lat": lat,
            "lon": lon,
            "temperature": temp,
            "aqi": aqi,
            "threat_status": threat_status,
            "ml_risk": risk,
            "rl_action": action
        })

    return results


# ---------------------------------------------------
# Serve Frontend
# ---------------------------------------------------
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")