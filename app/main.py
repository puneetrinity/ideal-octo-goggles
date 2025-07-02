
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvloop
import asyncio
from contextlib import asynccontextmanager

from app.api.ultra_fast_search import router as search_router, search_engine as api_search_engine
from app.search.ultra_fast_engine import UltraFastSearchEngine
from app.processing.batch_processor import MathematicalBatchProcessor
from app.logger import get_logger
from app.config import settings

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    # Initialize the search engine and batch processor
    search_engine = UltraFastSearchEngine(
        embedding_dim=settings.embedding_dim, 
        use_gpu=settings.use_gpu
    )
    batch_processor = MathematicalBatchProcessor()
    
    # Make the search engine available to the API router
    api_search_engine.search_engine = search_engine
    
    yield
    
    logger.info("Shutting down...")
    await batch_processor.shutdown()


app = FastAPI(
    title="Ultra-Fast Data Analysis System",
    description="A high-performance search system using advanced algorithms.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Ultra-Fast Data Analysis System"}

# Install uvloop for high-performance
uvloop.install()
