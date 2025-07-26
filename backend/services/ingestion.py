from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from typing import List, Optional
import structlog
import uuid

from database import (
    Project, LLMProviderModel, LLMLogModel, AppLogModel, SystemMetricModel
)
from models.schemas import (
    LLMLogCreate, AppLogCreate, SystemMetricCreate, IngestionResponse
)

logger = structlog.get_logger()

class IngestionService:
    """Service for handling data ingestion operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_or_create_project(self, api_key: str) -> Optional[Project]:
        """Get project by API key or return None if not found"""
        return self.db.query(Project).filter(Project.api_key == api_key).first()
    
    def get_or_create_provider(self, name: str, model_name: str, version: str = None) -> LLMProviderModel:
        """Get or create LLM provider"""
        provider = self.db.query(LLMProviderModel).filter(
            LLMProviderModel.name == name,
            LLMProviderModel.model_name == model_name,
            LLMProviderModel.version == version
        ).first()
        
        if not provider:
            provider = LLMProviderModel(
                name=name,
                model_name=model_name,
                version=version
            )
            self.db.add(provider)
            self.db.commit()
            self.db.refresh(provider)
            
        return provider
    
    def ingest_llm_log(self, log_data: LLMLogCreate) -> tuple[bool, str]:
        """Ingest a single LLM log entry"""
        try:
            # Get project
            project = self.get_or_create_project(log_data.project_api_key)
            if not project:
                return False, f"Invalid API key: {log_data.project_api_key}"
            
            # Get or create provider
            provider = self.get_or_create_provider(
                log_data.provider_name,
                log_data.model_name,
                getattr(log_data, 'version', None)
            )
            
            # Generate request ID if not provided
            request_id = log_data.request_id or str(uuid.uuid4())
            
            # Calculate latency if both timestamps are provided
            latency_ms = None
            if log_data.request_time and log_data.response_time:
                latency_ms = int((log_data.response_time - log_data.request_time).total_seconds() * 1000)
            
            # Create log entry
            llm_log = LLMLogModel(
                project_id=project.id,
                provider_id=provider.id,
                request_id=request_id,
                session_id=log_data.session_id,
                user_id=log_data.user_id,
                prompt=log_data.prompt,
                system_prompt=log_data.system_prompt,
                temperature=log_data.temperature,
                max_tokens=log_data.max_tokens,
                top_p=log_data.top_p,
                frequency_penalty=log_data.frequency_penalty,
                presence_penalty=log_data.presence_penalty,
                response=log_data.response,
                completion_tokens=log_data.completion_tokens,
                prompt_tokens=log_data.prompt_tokens,
                total_tokens=log_data.total_tokens,
                request_time=log_data.request_time or datetime.utcnow(),
                response_time=log_data.response_time,
                latency_ms=latency_ms,
                cost_usd=log_data.cost_usd,
                status=log_data.status.value,
                error_message=log_data.error_message,
                request_metadata=log_data.metadata
            )
            
            self.db.add(llm_log)
            self.db.commit()
            
            logger.info("LLM log ingested successfully", 
                       request_id=request_id, 
                       project_id=project.id)
            
            return True, "Success"
            
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to ingest LLM log", error=str(e), exc_info=True)
            return False, f"Database error: {str(e)}"
    
    def ingest_app_log(self, log_data: AppLogCreate) -> tuple[bool, str]:
        """Ingest a single application log entry"""
        try:
            # Get project
            project = self.get_or_create_project(log_data.project_api_key)
            if not project:
                return False, f"Invalid API key: {log_data.project_api_key}"
            
            # Create log entry
            app_log = AppLogModel(
                project_id=project.id,
                log_level=log_data.log_level.value,
                message=log_data.message,
                source=log_data.source,
                function_name=log_data.function_name,
                line_number=log_data.line_number,
                exception_type=log_data.exception_type,
                stack_trace=log_data.stack_trace,
                log_metadata=log_data.metadata,
                timestamp=log_data.timestamp or datetime.utcnow()
            )
            
            self.db.add(app_log)
            self.db.commit()
            
            logger.info("App log ingested successfully", 
                       project_id=project.id,
                       log_level=log_data.log_level.value)
            
            return True, "Success"
            
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to ingest app log", error=str(e), exc_info=True)
            return False, f"Database error: {str(e)}"
    
    def ingest_system_metric(self, metric_data: SystemMetricCreate) -> tuple[bool, str]:
        """Ingest a single system metric"""
        try:
            # Get project
            project = self.get_or_create_project(metric_data.project_api_key)
            if not project:
                return False, f"Invalid API key: {metric_data.project_api_key}"
            
            # Create metric entry
            system_metric = SystemMetricModel(
                project_id=project.id,
                metric_name=metric_data.metric_name,
                metric_value=metric_data.metric_value,
                metric_unit=metric_data.metric_unit,
                tags=metric_data.tags,
                timestamp=metric_data.timestamp or datetime.utcnow()
            )
            
            self.db.add(system_metric)
            self.db.commit()
            
            logger.info("System metric ingested successfully",
                       project_id=project.id,
                       metric_name=metric_data.metric_name)
            
            return True, "Success"
            
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to ingest system metric", error=str(e), exc_info=True)
            return False, f"Database error: {str(e)}"
    
    def bulk_ingest_llm_logs(self, logs: List[LLMLogCreate]) -> IngestionResponse:
        """Bulk ingest LLM logs"""
        successful = 0
        errors = []
        
        for i, log_data in enumerate(logs):
            success, message = self.ingest_llm_log(log_data)
            if success:
                successful += 1
            else:
                errors.append(f"Log {i}: {message}")
        
        return IngestionResponse(
            success=len(errors) == 0,
            message=f"Processed {successful}/{len(logs)} logs successfully",
            processed_count=successful,
            errors=errors
        )
    
    def bulk_ingest_app_logs(self, logs: List[AppLogCreate]) -> IngestionResponse:
        """Bulk ingest application logs"""
        successful = 0
        errors = []
        
        for i, log_data in enumerate(logs):
            success, message = self.ingest_app_log(log_data)
            if success:
                successful += 1
            else:
                errors.append(f"Log {i}: {message}")
        
        return IngestionResponse(
            success=len(errors) == 0,
            message=f"Processed {successful}/{len(logs)} logs successfully",
            processed_count=successful,
            errors=errors
        )
    
    def bulk_ingest_system_metrics(self, metrics: List[SystemMetricCreate]) -> IngestionResponse:
        """Bulk ingest system metrics"""
        successful = 0
        errors = []
        
        for i, metric_data in enumerate(metrics):
            success, message = self.ingest_system_metric(metric_data)
            if success:
                successful += 1
            else:
                errors.append(f"Metric {i}: {message}")
        
        return IngestionResponse(
            success=len(errors) == 0,
            message=f"Processed {successful}/{len(metrics)} metrics successfully",
            processed_count=successful,
            errors=errors
        )
