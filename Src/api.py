from fastapi import APIRouter, Depends, Query
from Src.ev_charging_bot import EVChargingStationBot

# Create a router
router = APIRouter()

# Initialize bot (or you can do this in server.py and pass it to router)
bot = EVChargingStationBot(
    file_path="Electric_Vehicle_Charging_Stations.csv",
    decoder_model="gpt2",
    use_decoder=False
)

session_memory = []


@router.get("/chat")
def chat(query: str = Query(..., description="User query for charging stations")):
    session_memory.append(f"From User: {query}")
    response = bot.chat_assistant(query)
    session_memory.append(f"From Bot: {response}")
    return {"query": query, "response": response['response']}

@router.get("/session")
def get_session():
    return {"session_memory": session_memory}

# @router.get("/nearby")
# def nearby(lat: float, lon: float, radius_km: float = 5):
#     response = bot.chat_find_nearby(lat, lon, radius_km)
#     return {"lat": lat, "lon": lon, "radius_km": radius_km, "response": response}

# @router.get("/fast-charging")
# def fast_charging(city: str):
#     response = bot.chat_find_fast_charging(city)
#     return {"city": city, "response": response}

# @router.get("/fast-charging")
# def fast_charging(city: str):
#     response = bot.chat_find_fast_charging(city)
#     return {"city": city, "response": response}
