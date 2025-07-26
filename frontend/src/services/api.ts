import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to:`, config.url);
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const apiService = {
  // Health check
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  // Projects
  async getProjects() {
    const response = await api.get('/api/v1/projects');
    return response.data;
  },

  async createProject(projectData: { name: string; description?: string }) {
    const response = await api.post('/api/v1/projects', projectData);
    return response.data;
  },

  // Analytics
  async getDashboardStats(apiKey: string, hours: number = 24) {
    const response = await api.get('/api/v1/analytics/dashboard', {
      params: { api_key: apiKey, hours }
    });
    return response.data;
  },

  async getTimeSeriesData(apiKey: string, hours: number = 24) {
    const response = await api.get('/api/v1/analytics/requests/timeseries', {
      params: { api_key: apiKey, hours }
    });
    return response.data;
  },

  async getProviderStats(apiKey: string, hours: number = 24) {
    const response = await api.get('/api/v1/analytics/providers', {
      params: { api_key: apiKey, hours }
    });
    return response.data;
  },

  async getRecentLogs(apiKey: string, logType: string = 'llm', limit: number = 100) {
    const response = await api.get('/api/v1/analytics/logs/recent', {
      params: { api_key: apiKey, log_type: logType, limit }
    });
    return response.data;
  },

  async getErrorAnalysis(apiKey: string, hours: number = 24) {
    const response = await api.get('/api/v1/analytics/errors', {
      params: { api_key: apiKey, hours }
    });
    return response.data;
  },

  async getCostBreakdown(apiKey: string, hours: number = 24) {
    const response = await api.get('/api/v1/analytics/cost/breakdown', {
      params: { api_key: apiKey, hours }
    });
    return response.data;
  },

  async getPerformanceMetrics(apiKey: string, hours: number = 24) {
    const response = await api.get('/api/v1/analytics/performance', {
      params: { api_key: apiKey, hours }
    });
    return response.data;
  },

  // Ingestion (for testing)
  async ingestLLMLog(logData: any) {
    const response = await api.post('/api/v1/ingest/llm', logData);
    return response.data;
  },

  async ingestAppLog(logData: any) {
    const response = await api.post('/api/v1/ingest/app', logData);
    return response.data;
  },

  async ingestSystemMetric(metricData: any) {
    const response = await api.post('/api/v1/ingest/metrics', metricData);
    return response.data;
  },

  // Bulk ingestion
  async bulkIngestLLMLogs(logs: any[]) {
    const response = await api.post('/api/v1/ingest/llm/bulk', { logs });
    return response.data;
  },

  async bulkIngestAppLogs(logs: any[]) {
    const response = await api.post('/api/v1/ingest/app/bulk', { logs });
    return response.data;
  },

  async bulkIngestSystemMetrics(metrics: any[]) {
    const response = await api.post('/api/v1/ingest/metrics/bulk', { metrics });
    return response.data;
  }
};

export default api;
