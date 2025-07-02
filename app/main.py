
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import sys
from contextlib import asynccontextmanager

# Only import uvloop on non-Windows systems
if sys.platform != "win32":
    import uvloop

from app.api.ultra_fast_search import router as search_router
from app.api import ultra_fast_search as api_module
from app.search.ultra_fast_engine import UltraFastSearchEngine
from app.processing.batch_processor import MathematicalBatchProcessor
from app.logger import get_enhanced_logger
from app.config import settings
from app.monitoring.health import HealthChecker

logger = get_enhanced_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Ultra-Fast Search System...")
    
    try:
        # Initialize the search engine and batch processor
        search_engine = UltraFastSearchEngine(
            embedding_dim=settings.embedding_dim, 
            use_gpu=settings.use_gpu
        )
        batch_processor = MathematicalBatchProcessor()
        
        # Initialize health checker
        health_checker_instance = HealthChecker(search_engine)
        
        # Make components available to the API router
        api_module.search_engine = search_engine
        api_module.health_checker = health_checker_instance
        
        # Start incremental indexing background processing
        await search_engine.incremental_manager.start_background_processing()
        
        logger.info("System startup completed successfully", extra_fields={
            'embedding_dim': settings.embedding_dim,
            'use_gpu': settings.use_gpu
        })
        
        yield
        
    except Exception as e:
        logger.error("Failed to start system", extra_fields={'error': str(e)})
        raise
    finally:
        logger.info("Shutting down...")
        
        try:
            # Cleanup incremental manager
            if hasattr(search_engine, 'incremental_manager'):
                await search_engine.incremental_manager.stop_background_processing()
            
            # Cleanup batch processor
            await batch_processor.shutdown()
            
            logger.info("System shutdown completed")
        except Exception as e:
            logger.error("Error during shutdown", extra_fields={'error': str(e)})


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

# Install uvloop for high-performance (Unix/Linux only)
if sys.platform != "win32":
    uvloop.install()
