"""
FastAPI REST API for CVAT-SAM3 Integration
Provides endpoints to analyze CVAT projects and tasks using SAM3.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cvat_client import CVATConfig
from cvat_sam3_service import (
    CVATSam3Service,
    ProjectAnalysisResult,
    TaskAnalysisResult,
    result_to_dict,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global service instance
_service: Optional[CVATSam3Service] = None

# Background task results storage
_task_results: Dict[str, Any] = {}


def get_cvat_config() -> CVATConfig:
    """Get CVAT configuration from environment variables."""
    return CVATConfig(
        server_url=os.getenv("CVAT_SERVER_URL", "http://192.168.50.15:8080"),
        username=os.getenv("CVAT_USERNAME", "david"),
        password=os.getenv("CVAT_PASSWORD", "a123321a"),
    )


def get_service() -> CVATSam3Service:
    """Get or create the service instance."""
    global _service
    if _service is None:
        config = get_cvat_config()
        threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
        _service = CVATSam3Service(config, confidence_threshold=threshold)
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting CVAT-SAM3 API service...")
    yield
    logger.info("Shutting down CVAT-SAM3 API service...")
    if _service:
        _service.close()


app = FastAPI(
    title="CVAT-SAM3 Object Detection API",
    description="API for analyzing CVAT projects and tasks using SAM3 model to detect objects",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class AnalyzeProjectRequest(BaseModel):
    project_id: int = Field(..., description="CVAT project ID")
    prompt: str = Field(..., description="Object to search for (e.g., 'a person', 'car')")
    max_frames_per_task: Optional[int] = Field(None, description="Max frames per task")
    max_tasks: Optional[int] = Field(None, description="Max tasks to analyze")
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    image_quality: str = Field("compressed", description="'original' or 'compressed'")


class AnalyzeTaskRequest(BaseModel):
    task_id: int = Field(..., description="CVAT task ID")
    prompt: str = Field(..., description="Object to search for")
    max_frames: Optional[int] = Field(None, description="Max frames to analyze")
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    image_quality: str = Field("compressed", description="'original' or 'compressed'")


class ProjectSummary(BaseModel):
    id: int
    name: str
    task_count: int


class TaskSummary(BaseModel):
    id: int
    name: str
    size: int
    status: str
    project_id: Optional[int]


class AnalysisStatus(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "CVAT-SAM3 Object Detection API"}


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


@app.get("/projects", response_model=List[ProjectSummary])
async def list_projects():
    """List all CVAT projects."""
    try:
        service = get_service()
        projects = service.cvat_client.get_projects()
        return [
            ProjectSummary(
                id=p["id"],
                name=p["name"],
                task_count=p.get("tasks", {}).get("count", 0) if isinstance(p.get("tasks"), dict) else 0
            )
            for p in projects
        ]
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_id}")
async def get_project(project_id: int):
    """Get project details."""
    try:
        service = get_service()
        project = service.cvat_client.get_project(project_id)
        return project
    except Exception as e:
        logger.error(f"Error getting project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_id}/tasks", response_model=List[TaskSummary])
async def list_project_tasks(project_id: int):
    """List all tasks in a project."""
    try:
        service = get_service()
        tasks = service.cvat_client.get_project_tasks(project_id)
        return [
            TaskSummary(
                id=t["id"],
                name=t["name"],
                size=t.get("size", 0),
                status=t.get("status", "unknown"),
                project_id=t.get("project_id")
            )
            for t in tasks
        ]
    except Exception as e:
        logger.error(f"Error listing tasks for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks/{task_id}")
async def get_task(task_id: int):
    """Get task details."""
    try:
        service = get_service()
        task = service.cvat_client.get_task(task_id)
        return task
    except Exception as e:
        logger.error(f"Error getting task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/task")
async def analyze_task(request: AnalyzeTaskRequest):
    """
    Analyze a single CVAT task for specified objects.

    This endpoint processes all frames in a task and returns detection results.
    """
    try:
        service = get_service()

        if request.confidence_threshold:
            service.confidence_threshold = request.confidence_threshold

        result = service.analyze_task(
            task_id=request.task_id,
            prompt=request.prompt,
            max_frames=request.max_frames,
            image_quality=request.image_quality
        )

        return result_to_dict(result)

    except Exception as e:
        logger.error(f"Error analyzing task {request.task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/project")
async def analyze_project(request: AnalyzeProjectRequest):
    """
    Analyze all tasks in a CVAT project for specified objects.

    This endpoint processes all frames in all tasks and returns detection results.
    Note: This can take a long time for large projects.
    """
    try:
        service = get_service()

        if request.confidence_threshold:
            service.confidence_threshold = request.confidence_threshold

        result = service.analyze_project(
            project_id=request.project_id,
            prompt=request.prompt,
            max_frames_per_task=request.max_frames_per_task,
            max_tasks=request.max_tasks,
            image_quality=request.image_quality
        )

        return result_to_dict(result)

    except Exception as e:
        logger.error(f"Error analyzing project {request.project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/project/{project_id}/frames-with-object")
async def get_frames_with_object(
    project_id: int,
    prompt: str = Query(..., description="Object to search for"),
    max_frames_per_task: Optional[int] = Query(None),
    confidence_threshold: Optional[float] = Query(None, ge=0.0, le=1.0)
):
    """
    Get list of frames that contain the specified object.

    Returns only the frames where the object was detected.
    """
    try:
        service = get_service()

        if confidence_threshold:
            service.confidence_threshold = confidence_threshold

        frames = service.get_frames_with_object(
            project_id=project_id,
            prompt=prompt,
            max_frames_per_task=max_frames_per_task
        )

        return {
            "project_id": project_id,
            "prompt": prompt,
            "total_frames_with_object": len(frames),
            "frames": [result_to_dict(f) for f in frames]
        }

    except Exception as e:
        logger.error(f"Error getting frames with object: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/project/{project_id}/frames-without-object")
async def get_frames_without_object(
    project_id: int,
    prompt: str = Query(..., description="Object to search for"),
    max_frames_per_task: Optional[int] = Query(None),
    confidence_threshold: Optional[float] = Query(None, ge=0.0, le=1.0)
):
    """
    Get list of frames that do NOT contain the specified object.

    Returns only the frames where the object was NOT detected.
    """
    try:
        service = get_service()

        if confidence_threshold:
            service.confidence_threshold = confidence_threshold

        frames = service.get_frames_without_object(
            project_id=project_id,
            prompt=prompt,
            max_frames_per_task=max_frames_per_task
        )

        return {
            "project_id": project_id,
            "prompt": prompt,
            "total_frames_without_object": len(frames),
            "frames": [result_to_dict(f) for f in frames]
        }

    except Exception as e:
        logger.error(f"Error getting frames without object: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/task/{task_id}/summary")
async def get_task_analysis_summary(
    task_id: int,
    prompt: str = Query(..., description="Object to search for"),
    max_frames: Optional[int] = Query(None),
    confidence_threshold: Optional[float] = Query(None, ge=0.0, le=1.0)
):
    """
    Get a summary of object detection for a task (without detailed frame results).
    """
    try:
        service = get_service()

        if confidence_threshold:
            service.confidence_threshold = confidence_threshold

        result = service.analyze_task(
            task_id=task_id,
            prompt=prompt,
            max_frames=max_frames
        )

        return {
            "task_id": result.task_id,
            "task_name": result.task_name,
            "prompt": prompt,
            "total_frames": result.total_frames,
            "frames_with_object": result.frames_with_object,
            "frames_without_object": result.frames_without_object,
            "detection_rate": result.detection_rate,
            "detection_rate_percent": f"{result.detection_rate:.1%}"
        }

    except Exception as e:
        logger.error(f"Error analyzing task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/project/{project_id}/summary")
async def get_project_analysis_summary(
    project_id: int,
    prompt: str = Query(..., description="Object to search for"),
    max_frames_per_task: Optional[int] = Query(None),
    max_tasks: Optional[int] = Query(None),
    confidence_threshold: Optional[float] = Query(None, ge=0.0, le=1.0)
):
    """
    Get a summary of object detection for a project (without detailed frame results).
    """
    try:
        service = get_service()

        if confidence_threshold:
            service.confidence_threshold = confidence_threshold

        result = service.analyze_project(
            project_id=project_id,
            prompt=prompt,
            max_frames_per_task=max_frames_per_task,
            max_tasks=max_tasks
        )

        task_summaries = [
            {
                "task_id": t.task_id,
                "task_name": t.task_name,
                "total_frames": t.total_frames,
                "frames_with_object": t.frames_with_object,
                "detection_rate": t.detection_rate
            }
            for t in result.task_results
        ]

        return {
            "project_id": result.project_id,
            "project_name": result.project_name,
            "prompt": prompt,
            "total_tasks": result.total_tasks,
            "total_frames": result.total_frames,
            "frames_with_object": result.frames_with_object,
            "overall_detection_rate": result.overall_detection_rate,
            "overall_detection_rate_percent": f"{result.overall_detection_rate:.1%}",
            "task_summaries": task_summaries
        }

    except Exception as e:
        logger.error(f"Error analyzing project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "cvat_api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )
