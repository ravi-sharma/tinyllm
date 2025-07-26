import React, { useState, useEffect } from 'react';
import { apiService } from '../services/api';

interface ProjectSelectorProps {
  selectedApiKey: string;
  onApiKeyChange: (apiKey: string) => void;
}

interface Project {
  id: number;
  name: string;
  description: string;
  api_key: string;
  created_at: string;
  updated_at: string;
}

const ProjectSelector: React.FC<ProjectSelectorProps> = ({ selectedApiKey, onApiKeyChange }) => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    const fetchProjects = async () => {
      try {
        setLoading(true);
        const projectsList = await apiService.getProjects();
        setProjects(projectsList);
        
        // Auto-select the first project if none is selected
        if (projectsList.length > 0 && !selectedApiKey) {
          onApiKeyChange(projectsList[0].api_key);
        }
      } catch (err) {
        setError('Failed to fetch projects');
        console.error('Project fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchProjects();
  }, [selectedApiKey, onApiKeyChange]);

  if (loading) {
    return (
      <div className="project-selector">
        <label>Project:</label>
        <select disabled>
          <option>Loading projects...</option>
        </select>
      </div>
    );
  }

  if (error) {
    return (
      <div className="project-selector">
        <label>Project:</label>
        <select disabled>
          <option>Error loading projects</option>
        </select>
      </div>
    );
  }

  return (
    <div className="project-selector">
      <label htmlFor="project-select">Project:</label>
      <select
        id="project-select"
        value={selectedApiKey}
        onChange={(e) => onApiKeyChange(e.target.value)}
      >
        <option value="">Select a project...</option>
        {projects.map((project) => (
          <option key={project.id} value={project.api_key}>
            {project.name} {project.description && `- ${project.description}`}
          </option>
        ))}
      </select>
    </div>
  );
};

export default ProjectSelector;
