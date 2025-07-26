import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.types import Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
import redis

# Database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://llm_user:llm_password@localhost:5432/llm_observability")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Database Models
class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text)
    api_key = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    llm_logs = relationship("LLMLogModel", back_populates="project")
    app_logs = relationship("AppLogModel", back_populates="project")
    system_metrics = relationship("SystemMetricModel", back_populates="project")

class LLMProviderModel(Base):
    __tablename__ = "llm_providers"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    model_name = Column(String, index=True)
    version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    llm_logs = relationship("LLMLogModel", back_populates="provider")

class LLMLogModel(Base):
    __tablename__ = "llm_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    provider_id = Column(Integer, ForeignKey("llm_providers.id"))
    request_id = Column(String, unique=True, index=True)
    session_id = Column(String, index=True)
    user_id = Column(String, index=True)
    
    # Request data
    prompt = Column(Text)
    system_prompt = Column(Text)
    temperature = Column(Numeric(3, 2))
    max_tokens = Column(Integer)
    top_p = Column(Numeric(3, 2))
    frequency_penalty = Column(Numeric(3, 2))
    presence_penalty = Column(Numeric(3, 2))
    
    # Response data
    response = Column(Text)
    completion_tokens = Column(Integer)
    prompt_tokens = Column(Integer)
    total_tokens = Column(Integer)
    
    # Timing and performance
    request_time = Column(DateTime, default=datetime.utcnow, index=True)
    response_time = Column(DateTime)
    latency_ms = Column(Integer)
    
    # Cost tracking
    cost_usd = Column(Numeric(10, 6))
    
    # Status and metadata
    status = Column(String, default="success", index=True)
    error_message = Column(Text)
    request_metadata = Column(JSONB)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    project = relationship("Project", back_populates="llm_logs")
    provider = relationship("LLMProviderModel", back_populates="llm_logs")

class AppLogModel(Base):
    __tablename__ = "app_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    log_level = Column(String, index=True)
    message = Column(Text)
    source = Column(String)
    function_name = Column(String)
    line_number = Column(Integer)
    exception_type = Column(String)
    stack_trace = Column(Text)
    log_metadata = Column(JSONB)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="app_logs")

class SystemMetricModel(Base):
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    metric_name = Column(String, index=True)
    metric_value = Column(Numeric(15, 6))
    metric_unit = Column(String)
    tags = Column(JSONB)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="system_metrics")

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    session_id = Column(String, unique=True, index=True)
    user_id = Column(String, index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    total_requests = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    total_cost = Column(Numeric(10, 6), default=0)
    session_metadata = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency to get Redis client
def get_redis():
    return redis_client

# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)
