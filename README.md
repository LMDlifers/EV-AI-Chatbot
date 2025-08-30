# EV AI Chatbot
Github repo for Maximally AI Shipathon Hackathon 2025


What is it?
Conversational Charger Locator: Go beyond a static map. Users should be able to ask in natural language.

Queries: "Where's the nearest fast charger?", "Find a CCS charger near my office," "Are there any free chargers at the downtown library?"
Filters: The chatbot should intelligently parse requests for:
Connector Type: NACS (Tesla), CCS, CHAdeMO
Charging Speed: Level 2 (AC) or Level 3 (DC Fast Charging)
Network: Electrify America, EVgo, ChargePoint, etc.
Cost: "Find free chargers nearby."

Startup Instructions (To change later)
cd to the base directory of the repo
Do pip install -r requirements.txt
Then, run python -m uvicorn Src.server:app --reload to start the backend
