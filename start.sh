#!/bin/bash

# LLM Observability Platform Startup Script

set -e

echo "🚀 Starting LLM Observability Platform..."
echo "========================================"

# Function to check if a service is ready
check_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts - waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service_name failed to start after $max_attempts attempts"
    return 1
}

# Start all services
echo "🐳 Starting Docker containers..."
docker-compose up -d

# Wait for services to be ready
check_service "http://localhost:8001/health" "Backend API"
check_service "http://localhost:3000" "Frontend Dashboard"

# Get the API key
echo ""
echo "🔑 Getting default project API key..."
API_KEY=$(curl -s http://localhost:8001/api/v1/projects | python3 -c "import sys, json; print(json.load(sys.stdin)[0]['api_key'])" 2>/dev/null || echo "")

if [ -n "$API_KEY" ]; then
    echo "✅ Default project API key: $API_KEY"
else
    echo "⚠️  Could not retrieve API key automatically"
fi

echo ""
echo "🎉 LLM Observability Platform is ready!"
echo "========================================"
echo ""
echo "🌐 Access Points:"
echo "   📊 Frontend Dashboard: http://localhost:3000"
echo "   📚 API Documentation: http://localhost:8001/docs"
echo "   🏥 Health Check: http://localhost:8001/health"
echo ""
echo "📋 Quick Test Commands:"
if [ -n "$API_KEY" ]; then
    echo "   # Test LLM log ingestion:"
    echo "   curl -X POST \"http://localhost:8001/api/v1/ingest/llm\" \\"
    echo "     -H \"Content-Type: application/json\" \\"
    echo "     -d '{"
    echo "       \"project_api_key\": \"$API_KEY\","
    echo "       \"provider_name\": \"openai\","
    echo "       \"model_name\": \"gpt-4\","
    echo "       \"prompt\": \"Hello, world!\","
    echo "       \"response\": \"Hello! How can I help you today?\","
    echo "       \"total_tokens\": 20,"
    echo "       \"cost_usd\": 0.0004,"
    echo "       \"status\": \"success\""
    echo "     }'"
    echo ""
    echo "   # Get dashboard analytics:"
    echo "   curl \"http://localhost:8001/api/v1/analytics/dashboard?api_key=$API_KEY\""
fi
echo ""
echo "   # Run comprehensive test suite:"
echo "   python3 test_platform.py"
echo ""
echo "🛑 To stop the platform:"
echo "   docker-compose down"
echo ""
echo "💡 Open http://localhost:3000 in your browser to see the dashboard!"
