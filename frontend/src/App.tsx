import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import ProjectSelector from './components/ProjectSelector';
import './App.css';

function App() {
  const [selectedApiKey, setSelectedApiKey] = React.useState<string>('');

  return (
    <div className="App">
      <Router>
        <header className="App-header">
          <div className="header-content">
            <h1>🔍 LLM Observability Platform</h1>
            <ProjectSelector 
              selectedApiKey={selectedApiKey} 
              onApiKeyChange={setSelectedApiKey} 
            />
          </div>
        </header>

        <main className="App-main">
          <Routes>
            <Route 
              path="/" 
              element={
                selectedApiKey ? (
                  <Dashboard apiKey={selectedApiKey} />
                ) : (
                  <div className="welcome-message">
                    <h2>Welcome to LLM Observability Platform</h2>
                    <p>Please select a project to view your dashboard.</p>
                  </div>
                )
              } 
            />
            <Route 
              path="/dashboard" 
              element={
                selectedApiKey ? (
                  <Dashboard apiKey={selectedApiKey} />
                ) : (
                  <Navigate to="/" replace />
                )
              } 
            />
          </Routes>
        </main>
      </Router>
    </div>
  );
}

export default App;
