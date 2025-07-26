import os
import time
import redis
import structlog
from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from database import DATABASE_URL, REDIS_URL

# Configure logging
logger = structlog.get_logger()

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Redis setup
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

class BackgroundWorker:
    """Background worker for processing analytics and maintenance tasks"""
    
    def __init__(self):
        self.running = True
        logger.info("Background worker initialized")
    
    def cleanup_old_logs(self):
        """Clean up old logs to prevent database bloat"""
        try:
            db = SessionLocal()
            
            # Delete logs older than 30 days
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            
            # Import here to avoid circular imports
            from database import LLMLogModel, AppLogModel, SystemMetricModel
            
            # Clean up LLM logs
            deleted_llm = db.query(LLMLogModel).filter(
                LLMLogModel.created_at < cutoff_date
            ).delete()
            
            # Clean up app logs
            deleted_app = db.query(AppLogModel).filter(
                AppLogModel.created_at < cutoff_date
            ).delete()
            
            # Clean up system metrics
            deleted_metrics = db.query(SystemMetricModel).filter(
                SystemMetricModel.created_at < cutoff_date
            ).delete()
            
            db.commit()
            
            logger.info("Cleanup completed", 
                       deleted_llm_logs=deleted_llm,
                       deleted_app_logs=deleted_app,
                       deleted_metrics=deleted_metrics)
            
        except Exception as e:
            logger.error("Cleanup failed", error=str(e), exc_info=True)
            db.rollback()
        finally:
            db.close()
    
    def update_cached_analytics(self):
        """Update cached analytics data for faster dashboard loading"""
        try:
            db = SessionLocal()
            
            # Import here to avoid circular imports
            from database import Project
            from services.analytics import AnalyticsService
            
            # Get all projects
            projects = db.query(Project).all()
            
            for project in projects:
                service = AnalyticsService(db)
                
                # Cache dashboard stats for different time ranges
                for hours in [1, 24, 168]:  # 1 hour, 1 day, 1 week
                    cache_key = f"dashboard_stats:{project.id}:{hours}"
                    
                    try:
                        stats = service.get_dashboard_stats(project.id, hours)
                        
                        # Store in Redis with 5 minute expiration
                        redis_client.setex(
                            cache_key,
                            300,  # 5 minutes
                            stats.json()
                        )
                        
                    except Exception as e:
                        logger.error("Failed to cache analytics", 
                                   project_id=project.id, 
                                   hours=hours,
                                   error=str(e))
            
            logger.info("Analytics cache updated", project_count=len(projects))
            
        except Exception as e:
            logger.error("Analytics caching failed", error=str(e), exc_info=True)
        finally:
            db.close()
    
    def process_alerts(self):
        """Process and trigger alerts based on conditions"""
        try:
            db = SessionLocal()
            
            # Import here to avoid circular imports
            from database import Project, LLMLogModel
            from sqlalchemy import func, and_
            
            # Get all projects
            projects = db.query(Project).all()
            
            for project in projects:
                # Check error rate in last hour
                one_hour_ago = datetime.utcnow() - timedelta(hours=1)
                
                total_requests = db.query(func.count(LLMLogModel.id)).filter(
                    and_(
                        LLMLogModel.project_id == project.id,
                        LLMLogModel.request_time >= one_hour_ago
                    )
                ).scalar() or 0
                
                if total_requests > 10:  # Only check if we have sufficient data
                    error_count = db.query(func.count(LLMLogModel.id)).filter(
                        and_(
                            LLMLogModel.project_id == project.id,
                            LLMLogModel.request_time >= one_hour_ago,
                            LLMLogModel.status != 'success'
                        )
                    ).scalar() or 0
                    
                    error_rate = (error_count / total_requests) * 100
                    
                    # Alert if error rate > 10%
                    if error_rate > 10:
                        alert_key = f"alert:error_rate:{project.id}"
                        
                        # Check if we already alerted recently
                        if not redis_client.exists(alert_key):
                            logger.warning("High error rate detected",
                                         project_id=project.id,
                                         project_name=project.name,
                                         error_rate=error_rate,
                                         total_requests=total_requests,
                                         error_count=error_count)
                            
                            # Set alert cooldown (1 hour)
                            redis_client.setex(alert_key, 3600, "triggered")
            
        except Exception as e:
            logger.error("Alert processing failed", error=str(e), exc_info=True)
        finally:
            db.close()
    
    def run(self):
        """Main worker loop"""
        logger.info("Background worker started")
        
        last_cleanup = datetime.utcnow()
        last_analytics_update = datetime.utcnow()
        last_alert_check = datetime.utcnow()
        
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                # Run cleanup daily
                if (current_time - last_cleanup).total_seconds() > 86400:  # 24 hours
                    self.cleanup_old_logs()
                    last_cleanup = current_time
                
                # Update analytics cache every 5 minutes
                if (current_time - last_analytics_update).total_seconds() > 300:  # 5 minutes
                    self.update_cached_analytics()
                    last_analytics_update = current_time
                
                # Check alerts every minute
                if (current_time - last_alert_check).total_seconds() > 60:  # 1 minute
                    self.process_alerts()
                    last_alert_check = current_time
                
                # Sleep for 30 seconds before next iteration
                time.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("Worker shutdown requested")
                self.running = False
            except Exception as e:
                logger.error("Worker error", error=str(e), exc_info=True)
                time.sleep(60)  # Wait a minute before retrying
        
        logger.info("Background worker stopped")

if __name__ == "__main__":
    worker = BackgroundWorker()
    worker.run()
