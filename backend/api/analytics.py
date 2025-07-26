from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any

from database import get_db, Project
from models.schemas import DashboardStats, TimeSeriesData, ProviderStats
from services.analytics import AnalyticsService

router = APIRouter()

def get_project_by_api_key(api_key: str, db: Session) -> Project:
    """Helper function to get project by API key"""
    project = db.query(Project).filter(Project.api_key == api_key).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@router.get("/analytics/dashboard", response_model=DashboardStats)
async def get_dashboard_stats(
    api_key: str = Query(..., description="Project API key"),
    hours: int = Query(24, description="Time range in hours"),
    db: Session = Depends(get_db)
):
    """Get overall dashboard statistics"""
    project = get_project_by_api_key(api_key, db)
    service = AnalyticsService(db)
    return service.get_dashboard_stats(project.id, hours)

@router.get("/analytics/requests/timeseries", response_model=List[TimeSeriesData])
async def get_requests_timeseries(
    api_key: str = Query(..., description="Project API key"),
    hours: int = Query(24, description="Time range in hours"),
    db: Session = Depends(get_db)
):
    """Get requests over time for charts"""
    project = get_project_by_api_key(api_key, db)
    service = AnalyticsService(db)
    return service.get_requests_time_series(project.id, hours)

@router.get("/analytics/providers", response_model=List[ProviderStats])
async def get_provider_stats(
    api_key: str = Query(..., description="Project API key"),
    hours: int = Query(24, description="Time range in hours"),
    db: Session = Depends(get_db)
):
    """Get statistics by provider/model"""
    project = get_project_by_api_key(api_key, db)
    service = AnalyticsService(db)
    return service.get_provider_stats(project.id, hours)

@router.get("/analytics/logs/recent")
async def get_recent_logs(
    api_key: str = Query(..., description="Project API key"),
    log_type: str = Query("llm", description="Type of logs (llm, app)"),
    limit: int = Query(100, description="Number of logs to return"),
    db: Session = Depends(get_db)
):
    """Get recent logs for display"""
    project = get_project_by_api_key(api_key, db)
    service = AnalyticsService(db)
    return service.get_recent_logs(project.id, limit, log_type)

@router.get("/analytics/errors")
async def get_error_analysis(
    api_key: str = Query(..., description="Project API key"),
    hours: int = Query(24, description="Time range in hours"),
    db: Session = Depends(get_db)
):
    """Get error analysis and patterns"""
    project = get_project_by_api_key(api_key, db)
    service = AnalyticsService(db)
    return service.get_error_analysis(project.id, hours)

@router.get("/analytics/cost/breakdown")
async def get_cost_breakdown(
    api_key: str = Query(..., description="Project API key"),
    hours: int = Query(24, description="Time range in hours"),
    db: Session = Depends(get_db)
):
    """Get cost breakdown by provider and time"""
    project = get_project_by_api_key(api_key, db)
    service = AnalyticsService(db)
    
    # This would expand to include more detailed cost analysis
    provider_stats = service.get_provider_stats(project.id, hours)
    
    return {
        "total_cost": sum(stat.total_cost for stat in provider_stats),
        "by_provider": [
            {
                "provider": stat.provider_name,
                "model": stat.model_name,
                "cost": stat.total_cost,
                "requests": stat.request_count
            }
            for stat in provider_stats
        ]
    }

@router.get("/analytics/performance")
async def get_performance_metrics(
    api_key: str = Query(..., description="Project API key"),
    hours: int = Query(24, description="Time range in hours"),
    db: Session = Depends(get_db)
):
    """Get performance metrics and trends"""
    project = get_project_by_api_key(api_key, db)
    service = AnalyticsService(db)
    
    dashboard_stats = service.get_dashboard_stats(project.id, hours)
    provider_stats = service.get_provider_stats(project.id, hours)
    
    return {
        "overall": {
            "avg_latency": dashboard_stats.avg_latency,
            "error_rate": dashboard_stats.error_rate,
            "total_requests": dashboard_stats.total_requests
        },
        "by_provider": [
            {
                "provider": stat.provider_name,
                "model": stat.model_name,
                "avg_latency": stat.avg_latency,
                "error_rate": stat.error_rate,
                "requests": stat.request_count
            }
            for stat in provider_stats
        ]
    }

@router.get("/analytics/usage/trends")
async def get_usage_trends(
    api_key: str = Query(..., description="Project API key"),
    hours: int = Query(168, description="Time range in hours (default 7 days)"),
    db: Session = Depends(get_db)
):
    """Get usage trends over time"""
    project = get_project_by_api_key(api_key, db)
    service = AnalyticsService(db)
    
    # Get time series data for requests
    requests_data = service.get_requests_time_series(project.id, hours)
    
    return {
        "requests_over_time": requests_data,
        "summary": {
            "total_data_points": len(requests_data),
            "time_range_hours": hours,
            "avg_requests_per_hour": sum(d.value for d in requests_data) / len(requests_data) if requests_data else 0
        }
    }
