
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import time

from app.search.ultra_fast_engine import UltraFastSearchEngine, SearchResult

import aiofiles

router = APIRouter(prefix="/api/v2", tags=["ultra-fast-search"])

# This will be set on application startup
search_engine: Optional[UltraFastSearchEngine] = None

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    num_results: int = Field(10, ge=1, le=100, description="Number of results to return")
    filters: Optional[Dict] = Field(None, description="Search filters")

class SearchResponse(BaseModel):
    success: bool
    results: List[Dict]
    total_found: int
    response_time_ms: float

@router.post("/search/ultra-fast", response_model=SearchResponse)
async def ultra_fast_search(request: SearchRequest):
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized.")

    start_time = time.time()
    try:
        results = await search_engine.search(
            query=request.query,
            num_results=request.num_results,
            filters=request.filters
        )
        response_time = (time.time() - start_time) * 1000

        formatted_results = [
            {
                "doc_id": r.doc_id,
                "similarity_score": r.similarity_score,
                "bm25_score": r.bm25_score,
                "combined_score": r.combined_score,
                **r.metadata
            }
            for r in results
        ]

        return SearchResponse(
            success=True,
            results=formatted_results,
            total_found=len(results),
            response_time_ms=response_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/performance", response_model=Dict)
async def get_search_performance():
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    return search_engine.get_performance_stats()

class IndexBuildRequest(BaseModel):
    data_source: str = Field(..., description="Path to the data file (e.g., data/resumes.json)")

@router.post("/admin/build-indexes")
async def build_indexes_endpoint(request: IndexBuildRequest, background_tasks: BackgroundTasks):
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized.")

    async def _build():
        import json
        async with aiofiles.open(request.data_source, mode='r') as f:
            content = await f.read()
            documents = json.loads(content)
        await search_engine.build_indexes(documents)

    background_tasks.add_task(_build)
    return {"message": "Index building started in the background."}
