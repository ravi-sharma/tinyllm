from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LLMStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"

# Project Models
class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass

class Project(ProjectBase):
    id: int
    api_key: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# LLM Provider Models
class LLMProviderBase(BaseModel):
    name: str
    model_name: str
    version: Optional[str] = None

class LLMProvider(LLMProviderBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# LLM Log Models
class LLMLogCreate(BaseModel):
    project_api_key: str
    provider_name: str
    model_name: str
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Request data
    prompt: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    
    # Response data
    response: str
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    
    # Timing
    request_time: Optional[datetime] = None
    response_time: Optional[datetime] = None
    latency_ms: Optional[int] = None
    
    # Cost
    cost_usd: Optional[Decimal] = None
    
    # Status
    status: LLMStatus = LLMStatus.SUCCESS
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class LLMLog(BaseModel):
    id: int
    project_id: int
    provider_id: int
    request_id: Optional[str]
    session_id: Optional[str]
    user_id: Optional[str]
    
    prompt: str
    system_prompt: Optional[str]
    temperature: Optional[Decimal]
    max_tokens: Optional[int]
    top_p: Optional[Decimal]
    frequency_penalty: Optional[Decimal]
    presence_penalty: Optional[Decimal]
    
    response: str
    completion_tokens: Optional[int]
    prompt_tokens: Optional[int]
    total_tokens: Optional[int]
    
    request_time: datetime
    response_time: Optional[datetime]
    latency_ms: Optional[int]
    
    cost_usd: Optional[Decimal]
    status: str
    error_message: Optional[str]
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        from_attributes = True

# Application Log Models
class AppLogCreate(BaseModel):
    project_api_key: str
    log_level: LogLevel
    message: str
    source: Optional[str] = None
    function_name: Optional[str] = None
    line_number: Optional[int] = None
    exception_type: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

class AppLog(BaseModel):
    id: int
    project_id: int
    log_level: str
    message: str
    source: Optional[str]
    function_name: Optional[str]
    line_number: Optional[int]
    exception_type: Optional[str]
    stack_trace: Optional[str]
    metadata: Optional[Dict[str, Any]]
    timestamp: datetime
    created_at: datetime
    
    class Config:
        from_attributes = True

# System Metrics Models
class SystemMetricCreate(BaseModel):
    project_api_key: str
    metric_name: str
    metric_value: float
    metric_unit: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

class SystemMetric(BaseModel):
    id: int
    project_id: int
    metric_name: str
    metric_value: Decimal
    metric_unit: Optional[str]
    tags: Optional[Dict[str, Any]]
    timestamp: datetime
    created_at: datetime
    
    class Config:
        from_attributes = True

# Dashboard Analytics Models
class DashboardStats(BaseModel):
    total_requests: int
    total_tokens: int
    total_cost: Decimal
    avg_latency: float
    error_rate: float
    requests_last_24h: int
    cost_last_24h: Decimal

class TimeSeriesData(BaseModel):
    timestamp: datetime
    value: float

class ProviderStats(BaseModel):
    provider_name: str
    model_name: str
    request_count: int
    total_tokens: int
    total_cost: Decimal
    avg_latency: float
    error_rate: float

# Bulk ingestion models
class BulkLLMLogs(BaseModel):
    logs: List[LLMLogCreate]

class BulkAppLogs(BaseModel):
    logs: List[AppLogCreate]

class BulkSystemMetrics(BaseModel):
    metrics: List[SystemMetricCreate]

# Response models
class IngestionResponse(BaseModel):
    success: bool
    message: str
    processed_count: int
    errors: List[str] = []

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    database: bool
    redis: bool
