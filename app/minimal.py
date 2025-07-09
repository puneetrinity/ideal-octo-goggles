#!/usr/bin/env python3
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Ultra-Fast Search System",
    description="High-performance search engine with RAG capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "Ultra-Fast Search System - Railway Edition",
        "version": "1.0.0",
        "status": "running",
        "port": os.getenv("PORT", "8000"),
        "environment": os.getenv("RAILWAY_ENVIRONMENT", "development")
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "ultra-fast-search",
        "version": "1.0.0"
    }

@app.get("/api/search")
def search(q: str = "", limit: int = 10):
    """Basic search endpoint"""
    return {
        "query": q,
        "results": [
            {
                "id": f"result_{i}",
                "content": f"Search result {i} for query: {q}",
                "score": 1.0 - (i * 0.1)
            }
            for i in range(min(limit, 5))
        ],
        "total": min(limit, 5),
        "message": "Basic search functionality"
    }

@app.get("/api/status")
def status():
    """System status endpoint"""
    return {
        "system": "operational",
        "search_engine": "basic",
        "features": ["basic_search", "health_check", "status"],
        "storage": {
            "status": "ready",
            "type": "ephemeral"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)