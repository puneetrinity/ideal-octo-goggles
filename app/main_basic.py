from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Basic Search API",
    description="A basic FastAPI service for Railway deployment",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Basic Search API is running on Railway!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Service is running",
        "service": "basic-search-api"
    }

@app.get("/test")
async def test():
    return {"test": "Railway deployment successful!"}