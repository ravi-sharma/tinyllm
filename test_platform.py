#!/usr/bin/env python3
"""
Test script for LLM Observability Platform
Demonstrates basic functionality including ingestion and analytics
"""

import requests
import json
import time
from datetime import datetime, timedelta
import random

BASE_URL = "http://localhost:8001/api/v1"

def get_default_project():
    """Get the default project API key"""
    response = requests.get(f"{BASE_URL}/projects")
    if response.status_code == 200:
        projects = response.json()
        if projects:
            return projects[0]['api_key']
    return None

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    response = requests.get("http://localhost:8001/health")
    if response.status_code == 200:
        health = response.json()
        print(f"✅ Health Status: {health['status']}")
        print(f"   Database: {'✅' if health['database'] else '❌'}")
        print(f"   Redis: {'✅' if health['redis'] else '❌'}")
        return True
    else:
        print("❌ Health check failed")
        return False

def test_llm_ingestion(api_key):
    """Test LLM log ingestion"""
    print("\n📝 Testing LLM log ingestion...")
    
    sample_logs = [
        {
            "provider_name": "openai",
            "model_name": "gpt-4",
            "prompt": "Write a haiku about programming",
            "response": "Code flows like water,\nLogic branches through my mind,\nBugs become features.",
            "total_tokens": 25,
            "cost_usd": 0.0005,
            "latency_ms": 1500,
            "status": "success"
        },
        {
            "provider_name": "anthropic",
            "model_name": "claude-3-sonnet",
            "prompt": "Explain quantum computing simply",
            "response": "Quantum computing uses quantum mechanics to process information...",
            "total_tokens": 150,
            "cost_usd": 0.003,
            "latency_ms": 2200,
            "status": "success"
        },
        {
            "provider_name": "openai",
            "model_name": "gpt-3.5-turbo",
            "prompt": "What's the weather like?",
            "response": "I don't have access to real-time weather data...",
            "total_tokens": 45,
            "cost_usd": 0.0001,
            "latency_ms": 800,
            "status": "success"
        }
    ]
    
    successful = 0
    for i, log in enumerate(sample_logs):
        log["project_api_key"] = api_key
        log["request_id"] = f"test-{int(time.time())}-{i}"
        
        response = requests.post(f"{BASE_URL}/ingest/llm", json=log)
        if response.status_code == 200:
            successful += 1
            print(f"   ✅ Log {i+1} ingested successfully")
        else:
            print(f"   ❌ Log {i+1} failed: {response.text}")
    
    print(f"📊 Ingested {successful}/{len(sample_logs)} logs successfully")
    return successful

def test_analytics(api_key):
    """Test analytics endpoints"""
    print("\n📊 Testing analytics...")
    
    # Dashboard stats
    response = requests.get(f"{BASE_URL}/analytics/dashboard", params={"api_key": api_key})
    if response.status_code == 200:
        stats = response.json()
        print("✅ Dashboard stats:")
        print(f"   Total requests: {stats['total_requests']}")
        print(f"   Total tokens: {stats['total_tokens']}")
        print(f"   Total cost: ${stats['total_cost']}")
        print(f"   Average latency: {stats['avg_latency']:.2f}ms")
        print(f"   Error rate: {stats['error_rate']:.2f}%")
    else:
        print("❌ Failed to get dashboard stats")
        return False
    
    # Recent logs
    response = requests.get(f"{BASE_URL}/analytics/logs/recent", params={"api_key": api_key, "limit": 5})
    if response.status_code == 200:
        logs = response.json()
        print(f"✅ Retrieved {len(logs)} recent logs")
        if logs:
            print("   Most recent log:")
            print(f"   - Prompt: {logs[0]['prompt'][:50]}...")
            print(f"   - Status: {logs[0]['status']}")
            print(f"   - Tokens: {logs[0]['total_tokens']}")
    else:
        print("❌ Failed to get recent logs")
    
    return True

def test_app_logs(api_key):
    """Test application log ingestion"""
    print("\n📋 Testing application log ingestion...")
    
    app_logs = [
        {
            "project_api_key": api_key,
            "log_level": "INFO",
            "message": "User authentication successful",
            "source": "auth_service",
            "function_name": "authenticate_user"
        },
        {
            "project_api_key": api_key,
            "log_level": "ERROR",
            "message": "Database connection timeout",
            "source": "database_service",
            "function_name": "connect_db",
            "exception_type": "TimeoutError"
        },
        {
            "project_api_key": api_key,
            "log_level": "WARNING",
            "message": "High memory usage detected",
            "source": "monitoring_service",
            "metadata": {"memory_usage": "85%", "threshold": "80%"}
        }
    ]
    
    successful = 0
    for i, log in enumerate(app_logs):
        response = requests.post(f"{BASE_URL}/ingest/app", json=log)
        if response.status_code == 200:
            successful += 1
            print(f"   ✅ App log {i+1} ingested successfully")
        else:
            print(f"   ❌ App log {i+1} failed: {response.text}")
    
    print(f"📊 Ingested {successful}/{len(app_logs)} app logs successfully")
    return successful

def test_system_metrics(api_key):
    """Test system metrics ingestion"""
    print("\n⚡ Testing system metrics ingestion...")
    
    metrics = [
        {
            "project_api_key": api_key,
            "metric_name": "cpu_usage",
            "metric_value": 75.5,
            "metric_unit": "percent",
            "tags": {"host": "server-1", "service": "llm-api"}
        },
        {
            "project_api_key": api_key,
            "metric_name": "memory_usage",
            "metric_value": 4.2,
            "metric_unit": "GB",
            "tags": {"host": "server-1", "service": "llm-api"}
        },
        {
            "project_api_key": api_key,
            "metric_name": "request_rate",
            "metric_value": 125.0,
            "metric_unit": "requests_per_minute",
            "tags": {"endpoint": "/api/v1/ingest/llm"}
        }
    ]
    
    successful = 0
    for i, metric in enumerate(metrics):
        response = requests.post(f"{BASE_URL}/ingest/metrics", json=metric)
        if response.status_code == 200:
            successful += 1
            print(f"   ✅ Metric {i+1} ingested successfully")
        else:
            print(f"   ❌ Metric {i+1} failed: {response.text}")
    
    print(f"📊 Ingested {successful}/{len(metrics)} metrics successfully")
    return successful

def main():
    """Run all tests"""
    print("🚀 LLM Observability Platform Test Suite")
    print("=" * 50)
    
    # Test health check
    if not test_health_check():
        print("❌ Platform is not healthy. Exiting.")
        return
    
    # Get API key
    api_key = get_default_project()
    if not api_key:
        print("❌ Could not get default project API key")
        return
    
    print(f"\n🔑 Using API Key: {api_key[:20]}...")
    
    # Run tests
    test_llm_ingestion(api_key)
    test_app_logs(api_key)
    test_system_metrics(api_key)
    
    # Wait a moment for data to be processed
    print("\n⏳ Waiting for data processing...")
    time.sleep(2)
    
    test_analytics(api_key)
    
    print("\n" + "=" * 50)
    print("🎉 Test suite completed!")
    print("\n🌐 Access points:")
    print("   Frontend Dashboard: http://localhost:3000")
    print("   API Documentation: http://localhost:8001/docs")
    print("   Health Check: http://localhost:8001/health")
    print("\n💡 Try opening the frontend dashboard to see your data!")

if __name__ == "__main__":
    main()
