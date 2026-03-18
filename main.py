# filepath: backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
from dotenv import load_dotenv
from routers import chat


# Robust log for debugging
current_dir = os.getcwd()
env_path = os.path.join(current_dir, ".env")
print(f"Loading env from: {env_path} (exists: {os.path.exists(env_path)})")
load_dotenv(env_path)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="RAG Chatbot API - NEC & Wattmonk")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# For development, we'll be broad with CORS to avoid port issues
allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
if "*" in allowed_origins:
    allowed_origins = ["*"]
print(f"CORS allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in allowed_origins else allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    print("Startup check completed. App is ready.")

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Backend is running"}
