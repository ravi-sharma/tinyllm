from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import uuid

from database import get_db, Project as ProjectModel
from models.schemas import Project, ProjectCreate

router = APIRouter()

@router.get("/projects", response_model=List[Project])
async def list_projects(db: Session = Depends(get_db)):
    """List all projects"""
    projects = db.query(ProjectModel).all()
    return projects

@router.post("/projects", response_model=Project)
async def create_project(
    project_data: ProjectCreate,
    db: Session = Depends(get_db)
):
    """Create a new project"""
    # Check if project name already exists
    existing = db.query(ProjectModel).filter(ProjectModel.name == project_data.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Project name already exists")
    
    # Generate API key
    api_key = f"llm-obs-{str(uuid.uuid4())}"
    
    # Create project
    project = ProjectModel(
        name=project_data.name,
        description=project_data.description,
        api_key=api_key
    )
    
    db.add(project)
    db.commit()
    db.refresh(project)
    
    return project

@router.get("/projects/{project_id}", response_model=Project)
async def get_project(
    project_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific project"""
    project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: int,
    db: Session = Depends(get_db)
):
    """Delete a project"""
    project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db.delete(project)
    db.commit()
    
    return {"message": "Project deleted successfully"}

@router.post("/projects/{project_id}/regenerate-key")
async def regenerate_api_key(
    project_id: int,
    db: Session = Depends(get_db)
):
    """Regenerate API key for a project"""
    project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Generate new API key
    new_api_key = f"llm-obs-{str(uuid.uuid4())}"
    project.api_key = new_api_key
    
    db.commit()
    db.refresh(project)
    
    return {"api_key": new_api_key, "message": "API key regenerated successfully"}
