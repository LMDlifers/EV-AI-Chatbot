# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from src.api import router
from contextlib import asynccontextmanager

HOST="0.0.0.0"
PORT= 8000

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting server with host {HOST} and port {PORT}")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)



if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
