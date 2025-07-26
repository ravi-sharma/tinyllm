-- LLM Observability Database Schema

-- Projects table to organize different applications/services
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    api_key VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- LLM providers/models
CREATE TABLE IF NOT EXISTS llm_providers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL, -- openai, anthropic, cohere, etc.
    model_name VARCHAR(100) NOT NULL, -- gpt-4, claude-3, etc.
    version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Main LLM requests/responses log
CREATE TABLE IF NOT EXISTS llm_logs (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    provider_id INTEGER REFERENCES llm_providers(id),
    request_id VARCHAR(255) UNIQUE,
    session_id VARCHAR(255),
    user_id VARCHAR(255),
    
    -- Request data
    prompt TEXT,
    system_prompt TEXT,
    temperature DECIMAL(3,2),
    max_tokens INTEGER,
    top_p DECIMAL(3,2),
    frequency_penalty DECIMAL(3,2),
    presence_penalty DECIMAL(3,2),
    
    -- Response data
    response TEXT,
    completion_tokens INTEGER,
    prompt_tokens INTEGER,
    total_tokens INTEGER,
    
    -- Timing and performance
    request_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    response_time TIMESTAMP WITH TIME ZONE,
    latency_ms INTEGER,
    
    -- Cost tracking
    cost_usd DECIMAL(10,6),
    
    -- Status and metadata
    status VARCHAR(50) DEFAULT 'success', -- success, error, timeout
    error_message TEXT,
    request_metadata JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Application logs (server logs, errors, etc.)
CREATE TABLE IF NOT EXISTS app_logs (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    log_level VARCHAR(20) NOT NULL, -- DEBUG, INFO, WARNING, ERROR, CRITICAL
    message TEXT NOT NULL,
    source VARCHAR(255), -- service name, file name, etc.
    function_name VARCHAR(255),
    line_number INTEGER,
    exception_type VARCHAR(255),
    stack_trace TEXT,
    log_metadata JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- System metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    metric_unit VARCHAR(50),
    tags JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User sessions for tracking user interactions
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255),
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    total_requests INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost DECIMAL(10,6) DEFAULT 0,
    session_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Alerts configuration
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    name VARCHAR(255) NOT NULL,
    alert_type VARCHAR(50) NOT NULL, -- error_rate, cost_threshold, latency, etc.
    condition_json JSONB NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Alert incidents
CREATE TABLE IF NOT EXISTS alert_incidents (
    id SERIAL PRIMARY KEY,
    alert_id INTEGER REFERENCES alerts(id),
    status VARCHAR(50) DEFAULT 'open', -- open, resolved, suppressed
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE,
    message TEXT,
    incident_metadata JSONB
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_llm_logs_project_id ON llm_logs(project_id);
CREATE INDEX IF NOT EXISTS idx_llm_logs_request_time ON llm_logs(request_time);
CREATE INDEX IF NOT EXISTS idx_llm_logs_session_id ON llm_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_llm_logs_status ON llm_logs(status);

CREATE INDEX IF NOT EXISTS idx_app_logs_project_id ON app_logs(project_id);
CREATE INDEX IF NOT EXISTS idx_app_logs_timestamp ON app_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_app_logs_log_level ON app_logs(log_level);

CREATE INDEX IF NOT EXISTS idx_system_metrics_project_id ON system_metrics(project_id);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name);

-- Insert default data
INSERT INTO projects (name, description, api_key) VALUES 
('default', 'Default project for testing', 'llm-obs-default-key-' || gen_random_uuid())
ON CONFLICT (name) DO NOTHING;

INSERT INTO llm_providers (name, model_name, version) VALUES 
('openai', 'gpt-4', '0613'),
('openai', 'gpt-4-turbo', '1106-preview'),
('openai', 'gpt-3.5-turbo', '0613'),
('anthropic', 'claude-3-opus', '20240229'),
('anthropic', 'claude-3-sonnet', '20240229'),
('anthropic', 'claude-3-haiku', '20240307')
ON CONFLICT DO NOTHING;
