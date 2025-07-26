from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List

from database import get_db
from models.schemas import (
    LLMLogCreate, AppLogCreate, SystemMetricCreate,
    BulkLLMLogs, BulkAppLogs, BulkSystemMetrics, IngestionResponse
)
from services.ingestion import IngestionService

router = APIRouter()

@router.post("/ingest/llm", response_model=IngestionResponse)
async def ingest_llm_log(
    log_data: LLMLogCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Ingest a single LLM log entry"""
    service = IngestionService(db)
    success, message = service.ingest_llm_log(log_data)
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    return IngestionResponse(
        success=True,
        message="LLM log ingested successfully",
        processed_count=1,
        errors=[]
    )

@router.post("/ingest/llm/bulk", response_model=IngestionResponse)
async def bulk_ingest_llm_logs(
    bulk_data: BulkLLMLogs,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Bulk ingest LLM logs"""
    if not bulk_data.logs:
        raise HTTPException(status_code=400, detail="No logs provided")
    
    service = IngestionService(db)
    result = service.bulk_ingest_llm_logs(bulk_data.logs)
    
    return result

@router.post("/ingest/app", response_model=IngestionResponse)
async def ingest_app_log(
    log_data: AppLogCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Ingest a single application log entry"""
    service = IngestionService(db)
    success, message = service.ingest_app_log(log_data)
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    return IngestionResponse(
        success=True,
        message="Application log ingested successfully",
        processed_count=1,
        errors=[]
    )

@router.post("/ingest/app/bulk", response_model=IngestionResponse)
async def bulk_ingest_app_logs(
    bulk_data: BulkAppLogs,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Bulk ingest application logs"""
    if not bulk_data.logs:
        raise HTTPException(status_code=400, detail="No logs provided")
    
    service = IngestionService(db)
    result = service.bulk_ingest_app_logs(bulk_data.logs)
    
    return result

@router.post("/ingest/metrics", response_model=IngestionResponse)
async def ingest_system_metric(
    metric_data: SystemMetricCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Ingest a single system metric"""
    service = IngestionService(db)
    success, message = service.ingest_system_metric(metric_data)
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    return IngestionResponse(
        success=True,
        message="System metric ingested successfully",
        processed_count=1,
        errors=[]
    )

@router.post("/ingest/metrics/bulk", response_model=IngestionResponse)
async def bulk_ingest_system_metrics(
    bulk_data: BulkSystemMetrics,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Bulk ingest system metrics"""
    if not bulk_data.metrics:
        raise HTTPException(status_code=400, detail="No metrics provided")
    
    service = IngestionService(db)
    result = service.bulk_ingest_system_metrics(bulk_data.metrics)
    
    return result

@router.get("/ingest/status")
async def ingestion_status():
    """Get ingestion service status"""
    return {
        "service": "ingestion",
        "status": "active",
        "endpoints": {
            "llm_single": "/ingest/llm",
            "llm_bulk": "/ingest/llm/bulk",
            "app_single": "/ingest/app",
            "app_bulk": "/ingest/app/bulk",
            "metrics_single": "/ingest/metrics",
            "metrics_bulk": "/ingest/metrics/bulk"
        }
    }
