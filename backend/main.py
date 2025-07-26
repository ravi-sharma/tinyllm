from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
import logging
import structlog
from typing import List, Optional

from database import get_db, get_redis, Project, LLMProviderModel
from models.schemas import (
    HealthCheck, LLMLogCreate, AppLogCreate, SystemMetricCreate,
    BulkLLMLogs, BulkAppLogs, BulkSystemMetrics, IngestionResponse,
    ProjectCreate, Project as ProjectSchema
)
from services.ingestion import IngestionService
from services.analytics import AnalyticsService
from api.ingestion import router as ingestion_router
from api.analytics import router as analytics_router
from api.projects import router as projects_router

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Create FastAPI app
app = FastAPI(
    title="LLM Observability Platform",
    description="A lightweight observability platform for monitoring LLM applications",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingestion_router, prefix="/api/v1", tags=["ingestion"])
app.include_router(analytics_router, prefix="/api/v1", tags=["analytics"])
app.include_router(projects_router, prefix="/api/v1", tags=["projects"])

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting LLM Observability Platform")
    
    # Create database tables if they don't exist
    from database import create_tables
    create_tables()
    
    logger.info("Application startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down LLM Observability Platform")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "LLM Observability Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint to verify system status"""
    try:
        # Check database connection
        db_healthy = True
        try:
            db.execute(text("SELECT 1"))
        except Exception as e:
            db_healthy = False
            logger.error("Database health check failed", error=str(e))

        # Check Redis connection
        redis_healthy = True
        try:
            redis_client = get_redis()
            redis_client.ping()
        except Exception as e:
            redis_healthy = False
            logger.error("Redis health check failed", error=str(e))

        overall_status = "healthy" if db_healthy and redis_healthy else "unhealthy"
        
        return HealthCheck(
            status=overall_status,
            timestamp=datetime.utcnow(),
            database=db_healthy,
            redis=redis_healthy
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/api/v1/status")
async def api_status():
    """API status endpoint with more detailed information"""
    return {
        "api_version": "1.0.0",
        "service": "llm-observability",
        "timestamp": datetime.utcnow(),
        "endpoints": {
            "ingestion": "/api/v1/ingest/",
            "analytics": "/api/v1/analytics/",
            "projects": "/api/v1/projects/"
        }
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(
        "Unhandled exception occurred",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True
    )
    return HTTPException(
        status_code=500,
        detail="Internal server error occurred"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
