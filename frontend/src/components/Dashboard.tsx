import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, BarChart, Bar, ResponsiveContainer } from 'recharts';
import { apiService } from '../services/api';

interface DashboardProps {
  apiKey: string;
}

interface DashboardStats {
  total_requests: number;
  total_tokens: number;
  total_cost: number;
  avg_latency: number;
  error_rate: number;
  requests_last_24h: number;
  cost_last_24h: number;
}

interface TimeSeriesData {
  timestamp: string;
  value: number;
}

interface RecentLog {
  id: number;
  request_id: string;
  prompt: string;
  response: string;
  status: string;
  latency_ms: number;
  total_tokens: number;
  cost_usd: number;
  created_at: string;
}

const Dashboard: React.FC<DashboardProps> = ({ apiKey }) => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData[]>([]);
  const [recentLogs, setRecentLogs] = useState<RecentLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError('');

        // Fetch dashboard stats
        const dashboardStats = await apiService.getDashboardStats(apiKey);
        setStats(dashboardStats);

        // Fetch time series data
        const timeSeries = await apiService.getTimeSeriesData(apiKey);
        setTimeSeriesData(timeSeries.map((item: any) => ({
          timestamp: new Date(item.timestamp).toLocaleTimeString(),
          value: item.value
        })));

        // Fetch recent logs
        const logs = await apiService.getRecentLogs(apiKey);
        setRecentLogs(logs);

      } catch (err) {
        setError('Failed to fetch dashboard data. Please check your API key.');
        console.error('Dashboard fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    if (apiKey) {
      fetchData();
      
      // Set up polling for real-time updates
      const interval = setInterval(fetchData, 30000); // Update every 30 seconds
      
      return () => clearInterval(interval);
    }
  }, [apiKey]);

  if (loading) {
    return (
      <div className="loading">
        <div className="loading-spinner"></div>
        Loading dashboard data...
      </div>
    );
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  if (!stats) {
    return <div className="error">No data available</div>;
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 4
    }).format(value);
  };

  const formatNumber = (value: number) => {
    return new Intl.NumberFormat('en-US').format(value);
  };

  const getStatusClassName = (status: string) => {
    switch (status) {
      case 'success':
        return 'status-success';
      case 'error':
        return 'status-error';
      case 'timeout':
        return 'status-timeout';
      default:
        return '';
    }
  };

  return (
    <div className="dashboard">
      {/* Stats Cards */}
      <div className="stats-grid">
        <div className="stat-card">
          <h3>Total Requests</h3>
          <p className="value">{formatNumber(stats.total_requests)}</p>
          <div className="change positive">
            +{formatNumber(stats.requests_last_24h)} last 24h
          </div>
        </div>

        <div className="stat-card">
          <h3>Total Tokens</h3>
          <p className="value">{formatNumber(stats.total_tokens)}</p>
        </div>

        <div className="stat-card">
          <h3>Total Cost</h3>
          <p className="value">{formatCurrency(stats.total_cost)}</p>
          <div className="change">
            {formatCurrency(stats.cost_last_24h)} last 24h
          </div>
        </div>

        <div className="stat-card">
          <h3>Avg Latency</h3>
          <p className="value">{Math.round(stats.avg_latency)}ms</p>
        </div>

        <div className="stat-card">
          <h3>Error Rate</h3>
          <p className="value">{stats.error_rate.toFixed(2)}%</p>
          <div className={`change ${stats.error_rate > 5 ? 'negative' : 'positive'}`}>
            {stats.error_rate > 5 ? 'High error rate' : 'Normal'}
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="charts-grid">
        <div className="chart-card">
          <h3>Requests Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#667eea" 
                strokeWidth={2}
                dot={{ fill: '#667eea' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3>Request Volume</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#667eea" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Logs */}
      <div className="logs-section">
        <div className="logs-header">
          <h3>Recent LLM Requests</h3>
          <span>{recentLogs.length} records</span>
        </div>
        
        <div className="logs-table-container">
          <table className="logs-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Request ID</th>
                <th>Prompt</th>
                <th>Response</th>
                <th>Status</th>
                <th>Latency</th>
                <th>Tokens</th>
                <th>Cost</th>
              </tr>
            </thead>
            <tbody>
              {recentLogs.slice(0, 20).map((log) => (
                <tr key={log.id}>
                  <td>{new Date(log.created_at).toLocaleString()}</td>
                  <td>
                    <code style={{ fontSize: '0.8rem' }}>
                      {log.request_id?.substring(0, 8)}...
                    </code>
                  </td>
                  <td style={{ maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {log.prompt}
                  </td>
                  <td style={{ maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {log.response}
                  </td>
                  <td>
                    <span className={getStatusClassName(log.status)}>
                      {log.status}
                    </span>
                  </td>
                  <td>{log.latency_ms ? `${log.latency_ms}ms` : '-'}</td>
                  <td>{formatNumber(log.total_tokens || 0)}</td>
                  <td>{formatCurrency(log.cost_usd || 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
