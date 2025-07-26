# LLM Observability Platform

A lightweight, containerized observability platform for monitoring LLM applications, similar to DeepEval but focused on real-time monitoring and analytics.

## Features

- **Real-time Monitoring**: Track LLM requests, responses, and performance metrics
- **Cost Tracking**: Monitor token usage and costs across different providers
- **Error Analysis**: Identify and analyze error patterns
- **Performance Metrics**: Latency, throughput, and success rate monitoring
- **Multi-Provider Support**: OpenAI, Anthropic, and other LLM providers
- **Dashboard**: Web-based dashboard with charts and analytics
- **Containerized**: Everything runs in Docker containers
- **Scalable**: PostgreSQL + Redis backend with background workers

## Quick Start

### Prerequisites

- Docker and Docker Compose
- At least 4GB of available RAM

### 1. Start the Platform

```bash
# Clone and start all services
git clone <your-repo>
cd llm-observability

# Start all services
docker-compose up -d

# View logs (optional)
docker-compose logs -f
```

### 2. Access the Dashboard

- **Frontend Dashboard**: http://localhost:3000
- **API Documentation**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health

### 3. Create a Project

The system will auto-create a "default" project, but you can create your own:

```bash
curl -X POST "http://localhost:8001/api/v1/projects" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app", "description": "My LLM Application"}'
```

### 4. Send Your First Log

```bash
# Get your API key from the dashboard or from the database
API_KEY="your-project-api-key"

curl -X POST "http://localhost:8001/api/v1/ingest/llm" \
  -H "Content-Type: application/json" \
  -d '{
    "project_api_key": "'$API_KEY'",
    "provider_name": "openai",
    "model_name": "gpt-4",
    "prompt": "What is the capital of France?",
    "response": "The capital of France is Paris.",
    "total_tokens": 15,
    "cost_usd": 0.0003,
    "latency_ms": 1200,
    "status": "success"
  }'
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React App     │    │   FastAPI       │    │   PostgreSQL    │
│   (Frontend)    │◄──►│   (Backend)     │◄──►│   (Database)    │
│   Port: 3000    │    │   Port: 8000    │    │   Port: 5432    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Background    │    │     Redis       │
                       │   Worker        │◄──►│   (Cache/Queue) │
                       │                 │    │   Port: 6379    │
                       └─────────────────┘    └─────────────────┘
```

## API Integration

### Python Example

```python
import requests
import time
from datetime import datetime

API_KEY = "your-project-api-key"
BASE_URL = "http://localhost:8001/api/v1"

def log_llm_request(prompt, response, provider="openai", model="gpt-4"):
    data = {
        "project_api_key": API_KEY,
        "provider_name": provider,
        "model_name": model,
        "prompt": prompt,
        "response": response,
        "request_time": datetime.utcnow().isoformat(),
        "response_time": datetime.utcnow().isoformat(),
        "total_tokens": len(prompt.split()) + len(response.split()),
        "status": "success"
    }
    
    response = requests.post(f"{BASE_URL}/ingest/llm", json=data)
    return response.json()

# Usage
log_llm_request(
    prompt="Explain quantum computing",
    response="Quantum computing uses quantum mechanics..."
)
```

### Node.js Example

```javascript
const axios = require('axios');

const API_KEY = 'your-project-api-key';
const BASE_URL = 'http://localhost:8001/api/v1';

async function logLLMRequest(prompt, response, provider = 'openai', model = 'gpt-4') {
  const data = {
    project_api_key: API_KEY,
    provider_name: provider,
    model_name: model,
    prompt: prompt,
    response: response,
    request_time: new Date().toISOString(),
    response_time: new Date().toISOString(),
    total_tokens: prompt.split(' ').length + response.split(' ').length,
    status: 'success'
  };

  try {
    const response = await axios.post(`${BASE_URL}/ingest/llm`, data);
    return response.data;
  } catch (error) {
    console.error('Failed to log request:', error);
  }
}

// Usage
logLLMRequest(
  'What is machine learning?',
  'Machine learning is a subset of artificial intelligence...'
);
```

## Monitoring Your Application

### Key Metrics

1. **Request Volume**: Total number of LLM requests
2. **Token Usage**: Input/output tokens consumed
3. **Cost Tracking**: Dollar amount spent across providers
4. **Latency**: Response time distribution
5. **Error Rate**: Failed requests percentage
6. **Provider Performance**: Comparison across different LLM providers

### Alerts (Coming Soon)

The platform includes basic alerting for:
- High error rates (>10%)
- Unusual cost spikes
- Performance degradation

## Data Management

### Retention Policy

- **Logs**: 30 days (configurable)
- **Metrics**: 90 days (configurable)
- **Analytics**: Real-time + cached summaries

### Backup

```bash
# Backup database
docker-compose exec postgres pg_dump -U llm_user llm_observability > backup.sql

# Restore database
docker-compose exec -T postgres psql -U llm_user llm_observability < backup.sql
```

## Development

### Backend Development

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run locally (requires PostgreSQL and Redis)
DATABASE_URL="postgresql://llm_user:llm_password@localhost:5432/llm_observability" \
REDIS_URL="redis://localhost:6379" \
uvicorn main:app --reload
```

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Start development server
REACT_APP_API_URL="http://localhost:8001" npm start
```

### Adding New Features

1. **New Log Type**: Add to `models/schemas.py` and create ingestion endpoint
2. **New Metric**: Add to analytics service and frontend components
3. **New Provider**: Add to database seed data and update provider stats

## Configuration

### Environment Variables

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `ENVIRONMENT`: development/production
- `REACT_APP_API_URL`: Frontend API endpoint

### Scaling

For production use:

1. **Database**: Use managed PostgreSQL (AWS RDS, Google Cloud SQL)
2. **Redis**: Use managed Redis (AWS ElastiCache, Redis Cloud)
3. **Load Balancer**: Add nginx for multiple backend instances
4. **Monitoring**: Add Prometheus/Grafana for infrastructure metrics

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   docker-compose logs postgres
   docker-compose restart postgres
   ```

2. **Frontend Can't Connect to Backend**
   - Check API URL in frontend environment
   - Verify backend is running on port 8000

3. **High Memory Usage**
   - Reduce log retention period
   - Scale up database resources

### Logs

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs backend
docker-compose logs frontend
docker-compose logs postgres
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- **Issues**: GitHub Issues
- **Documentation**: This README
- **API Docs**: http://localhost:8001/docs (when running)

---

Built with ❤️ for the LLM community
