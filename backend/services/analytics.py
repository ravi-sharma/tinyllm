from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from decimal import Decimal
import structlog

from database import (
    Project, LLMProviderModel, LLMLogModel, AppLogModel, SystemMetricModel
)
from models.schemas import (
    DashboardStats, TimeSeriesData, ProviderStats
)

logger = structlog.get_logger()

class AnalyticsService:
    """Service for analytics and dashboard data"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_dashboard_stats(self, project_id: int, hours: int = 24) -> DashboardStats:
        """Get overall dashboard statistics"""
        try:
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            # Total requests
            total_requests = self.db.query(func.count(LLMLogModel.id)).filter(
                LLMLogModel.project_id == project_id
            ).scalar() or 0
            
            # Total tokens
            total_tokens = self.db.query(func.sum(LLMLogModel.total_tokens)).filter(
                LLMLogModel.project_id == project_id
            ).scalar() or 0
            
            # Total cost
            total_cost = self.db.query(func.sum(LLMLogModel.cost_usd)).filter(
                LLMLogModel.project_id == project_id
            ).scalar() or Decimal('0.00')
            
            # Average latency
            avg_latency = self.db.query(func.avg(LLMLogModel.latency_ms)).filter(
                and_(
                    LLMLogModel.project_id == project_id,
                    LLMLogModel.latency_ms.isnot(None)
                )
            ).scalar() or 0.0
            
            # Error rate
            total_with_status = self.db.query(func.count(LLMLogModel.id)).filter(
                LLMLogModel.project_id == project_id
            ).scalar() or 1
            
            error_count = self.db.query(func.count(LLMLogModel.id)).filter(
                and_(
                    LLMLogModel.project_id == project_id,
                    LLMLogModel.status != 'success'
                )
            ).scalar() or 0
            
            error_rate = (error_count / total_with_status) * 100 if total_with_status > 0 else 0.0
            
            # Last 24h requests
            requests_24h = self.db.query(func.count(LLMLogModel.id)).filter(
                and_(
                    LLMLogModel.project_id == project_id,
                    LLMLogModel.request_time >= start_time
                )
            ).scalar() or 0
            
            # Last 24h cost
            cost_24h = self.db.query(func.sum(LLMLogModel.cost_usd)).filter(
                and_(
                    LLMLogModel.project_id == project_id,
                    LLMLogModel.request_time >= start_time
                )
            ).scalar() or Decimal('0.00')
            
            return DashboardStats(
                total_requests=total_requests,
                total_tokens=total_tokens,
                total_cost=total_cost,
                avg_latency=float(avg_latency),
                error_rate=error_rate,
                requests_last_24h=requests_24h,
                cost_last_24h=cost_24h
            )
            
        except Exception as e:
            logger.error("Failed to get dashboard stats", error=str(e), exc_info=True)
            return DashboardStats(
                total_requests=0,
                total_tokens=0,
                total_cost=Decimal('0.00'),
                avg_latency=0.0,
                error_rate=0.0,
                requests_last_24h=0,
                cost_last_24h=Decimal('0.00')
            )
    
    def get_requests_time_series(self, project_id: int, hours: int = 24) -> List[TimeSeriesData]:
        """Get requests over time for charts"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            # Group by hour
            results = self.db.query(
                func.date_trunc('hour', LLMLogModel.request_time).label('hour'),
                func.count(LLMLogModel.id).label('count')
            ).filter(
                and_(
                    LLMLogModel.project_id == project_id,
                    LLMLogModel.request_time >= start_time
                )
            ).group_by('hour').order_by('hour').all()
            
            return [
                TimeSeriesData(timestamp=result.hour, value=float(result.count))
                for result in results
            ]
            
        except Exception as e:
            logger.error("Failed to get time series data", error=str(e), exc_info=True)
            return []
    
    def get_provider_stats(self, project_id: int, hours: int = 24) -> List[ProviderStats]:
        """Get statistics by provider/model"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            results = self.db.query(
                LLMProviderModel.name.label('provider_name'),
                LLMProviderModel.model_name.label('model_name'),
                func.count(LLMLogModel.id).label('request_count'),
                func.sum(LLMLogModel.total_tokens).label('total_tokens'),
                func.sum(LLMLogModel.cost_usd).label('total_cost'),
                func.avg(LLMLogModel.latency_ms).label('avg_latency'),
                func.count(LLMLogModel.id).filter(LLMLogModel.status != 'success').label('error_count')
            ).join(
                LLMProviderModel, LLMLogModel.provider_id == LLMProviderModel.id
            ).filter(
                and_(
                    LLMLogModel.project_id == project_id,
                    LLMLogModel.request_time >= start_time
                )
            ).group_by(
                LLMProviderModel.name, LLMProviderModel.model_name
            ).all()
            
            provider_stats = []
            for result in results:
                error_rate = (float(result.error_count) / float(result.request_count)) * 100 if result.request_count > 0 else 0.0
                
                provider_stats.append(ProviderStats(
                    provider_name=result.provider_name,
                    model_name=result.model_name,
                    request_count=result.request_count,
                    total_tokens=result.total_tokens or 0,
                    total_cost=result.total_cost or Decimal('0.00'),
                    avg_latency=float(result.avg_latency) if result.avg_latency else 0.0,
                    error_rate=error_rate
                ))
            
            return provider_stats
            
        except Exception as e:
            logger.error("Failed to get provider stats", error=str(e), exc_info=True)
            return []
    
    def get_recent_logs(self, project_id: int, limit: int = 100, log_type: str = "llm") -> List[Dict[str, Any]]:
        """Get recent logs for display"""
        try:
            if log_type == "llm":
                results = self.db.query(LLMLogModel).filter(
                    LLMLogModel.project_id == project_id
                ).order_by(desc(LLMLogModel.created_at)).limit(limit).all()
                
                return [
                    {
                        "id": log.id,
                        "request_id": log.request_id,
                        "prompt": log.prompt[:100] + "..." if len(log.prompt) > 100 else log.prompt,
                        "response": log.response[:100] + "..." if len(log.response) > 100 else log.response,
                        "status": log.status,
                        "latency_ms": log.latency_ms,
                        "total_tokens": log.total_tokens,
                        "cost_usd": float(log.cost_usd) if log.cost_usd else 0.0,
                        "created_at": log.created_at
                    }
                    for log in results
                ]
            
            elif log_type == "app":
                results = self.db.query(AppLogModel).filter(
                    AppLogModel.project_id == project_id
                ).order_by(desc(AppLogModel.created_at)).limit(limit).all()
                
                return [
                    {
                        "id": log.id,
                        "log_level": log.log_level,
                        "message": log.message,
                        "source": log.source,
                        "timestamp": log.timestamp,
                        "created_at": log.created_at
                    }
                    for log in results
                ]
            
            return []
            
        except Exception as e:
            logger.error("Failed to get recent logs", error=str(e), exc_info=True)
            return []
    
    def get_error_analysis(self, project_id: int, hours: int = 24) -> Dict[str, Any]:
        """Get error analysis and patterns"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            # Error count by type
            error_types = self.db.query(
                LLMLogModel.status,
                func.count(LLMLogModel.id).label('count')
            ).filter(
                and_(
                    LLMLogModel.project_id == project_id,
                    LLMLogModel.status != 'success',
                    LLMLogModel.request_time >= start_time
                )
            ).group_by(LLMLogModel.status).all()
            
            # Recent errors
            recent_errors = self.db.query(LLMLogModel).filter(
                and_(
                    LLMLogModel.project_id == project_id,
                    LLMLogModel.status != 'success',
                    LLMLogModel.request_time >= start_time
                )
            ).order_by(desc(LLMLogModel.created_at)).limit(10).all()
            
            return {
                "error_types": [
                    {"status": error.status, "count": error.count}
                    for error in error_types
                ],
                "recent_errors": [
                    {
                        "request_id": error.request_id,
                        "status": error.status,
                        "error_message": error.error_message,
                        "created_at": error.created_at
                    }
                    for error in recent_errors
                ]
            }
            
        except Exception as e:
            logger.error("Failed to get error analysis", error=str(e), exc_info=True)
            return {"error_types": [], "recent_errors": []}
