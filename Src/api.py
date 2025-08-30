from fastapi import APIRouter, Query
from src.ev_charging_bot import EVChargingStationBot

# Create a router
router = APIRouter()

# Initialize bot (or you can do this in server.py and pass it to router)
bot = EVChargingStationBot(
    file_path="Electric_Vehicle_Charging_Stations.csv",
    decoder_model="gpt2",
    use_decoder=True
)

@router.get("/chat")
def chat(query: str = Query(..., description="User query for charging stations")):
    response = bot.chat_semantic_search(query)
    return {"query": query, "response": response}

@router.get("/nearby")
def nearby(lat: float, lon: float, radius_km: float = 5):
    response = bot.chat_find_nearby(lat, lon, radius_km)
    return {"lat": lat, "lon": lon, "radius_km": radius_km, "response": response}

@router.get("/fast-charging")
def fast_charging(city: str):
    response = bot.chat_find_fast_charging(city)
    return {"city": city, "response": response}