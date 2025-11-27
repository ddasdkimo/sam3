"""
CVAT-SAM3 Integration Service
Uses SAM3 model to analyze images from CVAT and detect specified objects.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from cvat_client import CVATClient, CVATConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lazy load SAM3 model to avoid import issues when not needed
_image_model = None
_image_processor = None


def get_sam3_model():
    """Lazy load SAM3 model."""
    global _image_model, _image_processor

    if _image_model is None:
        logger.info("Loading SAM3 model...")
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _image_model = build_sam3_image_model(
            device=device,
            eval_mode=True,
            load_from_HF=True,
        )
        _image_processor = Sam3Processor(_image_model, device=device)
        logger.info(f"SAM3 model loaded on {device}")

    return _image_model, _image_processor


@dataclass
class DetectionResult:
    """Result of object detection on a single image."""
    task_id: int
    task_name: str
    frame_number: int
    frame_name: Optional[str]
    object_detected: bool
    num_objects: int
    confidence_scores: List[float]
    prompt: str
    error: Optional[str] = None


@dataclass
class TaskAnalysisResult:
    """Result of analyzing all frames in a task."""
    task_id: int
    task_name: str
    total_frames: int
    frames_with_object: int
    frames_without_object: int
    detection_rate: float
    frame_results: List[DetectionResult] = field(default_factory=list)


@dataclass
class ProjectAnalysisResult:
    """Result of analyzing all tasks in a project."""
    project_id: int
    project_name: str
    prompt: str
    total_tasks: int
    total_frames: int
    frames_with_object: int
    overall_detection_rate: float
    task_results: List[TaskAnalysisResult] = field(default_factory=list)


class CVATSam3Service:
    """Service for analyzing CVAT projects/tasks using SAM3."""

    def __init__(
        self,
        cvat_config: CVATConfig,
        confidence_threshold: float = 0.5,
        load_model_on_init: bool = False
    ):
        """
        Initialize the service.

        Args:
            cvat_config: CVAT server configuration
            confidence_threshold: Minimum confidence score for detection
            load_model_on_init: Whether to load SAM3 model immediately
        """
        self.cvat_client = CVATClient(cvat_config)
        self.confidence_threshold = confidence_threshold

        if load_model_on_init:
            get_sam3_model()

    def detect_object_in_image(
        self,
        image: Image.Image,
        prompt: str,
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detect objects in a single image using SAM3.

        Args:
            image: PIL Image to analyze
            prompt: Text prompt describing the object to detect
            confidence_threshold: Override default confidence threshold

        Returns:
            Dictionary with detection results
        """
        model, processor = get_sam3_model()
        threshold = confidence_threshold or self.confidence_threshold

        try:
            # Set confidence threshold
            processor.confidence_threshold = threshold

            # Process image
            state = processor.set_image(image)
            output = processor.set_text_prompt(prompt=prompt, state=state)

            # Get results
            masks = output.get("masks", None)
            scores = output.get("scores", None)

            if masks is None or len(masks) == 0:
                return {
                    "detected": False,
                    "num_objects": 0,
                    "scores": [],
                    "masks": None
                }

            # Convert tensors to numpy
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

            # Filter by confidence threshold
            if scores is not None:
                valid_indices = scores >= threshold
                if isinstance(valid_indices, np.ndarray):
                    valid_indices = valid_indices.flatten()
                    masks = masks[valid_indices]
                    scores = scores[valid_indices]

            num_objects = len(masks) if masks is not None else 0

            return {
                "detected": num_objects > 0,
                "num_objects": num_objects,
                "scores": scores.tolist() if scores is not None and num_objects > 0 else [],
                "masks": masks
            }

        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return {
                "detected": False,
                "num_objects": 0,
                "scores": [],
                "masks": None,
                "error": str(e)
            }

    def analyze_task(
        self,
        task_id: int,
        prompt: str,
        max_frames: Optional[int] = None,
        image_quality: str = "compressed"
    ) -> TaskAnalysisResult:
        """
        Analyze all frames in a CVAT task.

        Args:
            task_id: CVAT task ID
            prompt: Object to search for
            max_frames: Maximum frames to analyze (None for all)
            image_quality: 'original' or 'compressed'

        Returns:
            TaskAnalysisResult with detection results for all frames
        """
        task = self.cvat_client.get_task(task_id)
        task_name = task.get("name", f"Task {task_id}")
        total_frames = task.get("size", 0)

        logger.info(f"Analyzing task '{task_name}' ({task_id}) with {total_frames} frames for: {prompt}")

        # Get frame metadata
        try:
            meta = self.cvat_client.get_task_data_meta(task_id)
            frames_info = meta.get("frames", [])
        except Exception:
            frames_info = []

        if max_frames:
            total_frames = min(total_frames, max_frames)

        frame_results = []
        frames_with_object = 0

        for frame_num in range(total_frames):
            try:
                # Download frame
                image = self.cvat_client.get_task_frame(task_id, frame_num, image_quality)

                # Get frame name if available
                frame_name = None
                if frame_num < len(frames_info):
                    frame_name = frames_info[frame_num].get("name")

                # Detect objects
                detection = self.detect_object_in_image(image, prompt)

                result = DetectionResult(
                    task_id=task_id,
                    task_name=task_name,
                    frame_number=frame_num,
                    frame_name=frame_name,
                    object_detected=detection["detected"],
                    num_objects=detection["num_objects"],
                    confidence_scores=detection["scores"],
                    prompt=prompt,
                    error=detection.get("error")
                )

                frame_results.append(result)

                if detection["detected"]:
                    frames_with_object += 1

                logger.debug(
                    f"Frame {frame_num + 1}/{total_frames}: "
                    f"{'Found' if detection['detected'] else 'Not found'} "
                    f"({detection['num_objects']} objects)"
                )

            except Exception as e:
                logger.warning(f"Error analyzing frame {frame_num}: {e}")
                frame_results.append(DetectionResult(
                    task_id=task_id,
                    task_name=task_name,
                    frame_number=frame_num,
                    frame_name=None,
                    object_detected=False,
                    num_objects=0,
                    confidence_scores=[],
                    prompt=prompt,
                    error=str(e)
                ))

        detection_rate = frames_with_object / total_frames if total_frames > 0 else 0

        logger.info(
            f"Task '{task_name}' analysis complete: "
            f"{frames_with_object}/{total_frames} frames contain '{prompt}' "
            f"({detection_rate:.1%})"
        )

        return TaskAnalysisResult(
            task_id=task_id,
            task_name=task_name,
            total_frames=total_frames,
            frames_with_object=frames_with_object,
            frames_without_object=total_frames - frames_with_object,
            detection_rate=detection_rate,
            frame_results=frame_results
        )

    def analyze_project(
        self,
        project_id: int,
        prompt: str,
        max_frames_per_task: Optional[int] = None,
        max_tasks: Optional[int] = None,
        image_quality: str = "compressed"
    ) -> ProjectAnalysisResult:
        """
        Analyze all tasks in a CVAT project.

        Args:
            project_id: CVAT project ID
            prompt: Object to search for
            max_frames_per_task: Maximum frames per task (None for all)
            max_tasks: Maximum tasks to analyze (None for all)
            image_quality: 'original' or 'compressed'

        Returns:
            ProjectAnalysisResult with detection results for all tasks
        """
        project = self.cvat_client.get_project(project_id)
        project_name = project.get("name", f"Project {project_id}")
        tasks = self.cvat_client.get_project_tasks(project_id)

        if max_tasks:
            tasks = tasks[:max_tasks]

        logger.info(
            f"Analyzing project '{project_name}' ({project_id}) "
            f"with {len(tasks)} tasks for: {prompt}"
        )

        task_results = []
        total_frames = 0
        frames_with_object = 0

        for i, task in enumerate(tasks):
            task_id = task["id"]
            logger.info(f"Processing task {i + 1}/{len(tasks)}: {task.get('name', task_id)}")

            task_result = self.analyze_task(
                task_id=task_id,
                prompt=prompt,
                max_frames=max_frames_per_task,
                image_quality=image_quality
            )

            task_results.append(task_result)
            total_frames += task_result.total_frames
            frames_with_object += task_result.frames_with_object

        overall_detection_rate = frames_with_object / total_frames if total_frames > 0 else 0

        logger.info(
            f"Project '{project_name}' analysis complete: "
            f"{frames_with_object}/{total_frames} frames contain '{prompt}' "
            f"({overall_detection_rate:.1%})"
        )

        return ProjectAnalysisResult(
            project_id=project_id,
            project_name=project_name,
            prompt=prompt,
            total_tasks=len(tasks),
            total_frames=total_frames,
            frames_with_object=frames_with_object,
            overall_detection_rate=overall_detection_rate,
            task_results=task_results
        )

    def get_frames_with_object(
        self,
        project_id: int,
        prompt: str,
        max_frames_per_task: Optional[int] = None
    ) -> List[DetectionResult]:
        """
        Get list of frames that contain the specified object.

        Args:
            project_id: CVAT project ID
            prompt: Object to search for
            max_frames_per_task: Maximum frames per task

        Returns:
            List of DetectionResult for frames containing the object
        """
        result = self.analyze_project(
            project_id=project_id,
            prompt=prompt,
            max_frames_per_task=max_frames_per_task
        )

        positive_frames = []
        for task_result in result.task_results:
            for frame_result in task_result.frame_results:
                if frame_result.object_detected:
                    positive_frames.append(frame_result)

        return positive_frames

    def get_frames_without_object(
        self,
        project_id: int,
        prompt: str,
        max_frames_per_task: Optional[int] = None
    ) -> List[DetectionResult]:
        """
        Get list of frames that do NOT contain the specified object.

        Args:
            project_id: CVAT project ID
            prompt: Object to search for
            max_frames_per_task: Maximum frames per task

        Returns:
            List of DetectionResult for frames NOT containing the object
        """
        result = self.analyze_project(
            project_id=project_id,
            prompt=prompt,
            max_frames_per_task=max_frames_per_task
        )

        negative_frames = []
        for task_result in result.task_results:
            for frame_result in task_result.frame_results:
                if not frame_result.object_detected:
                    negative_frames.append(frame_result)

        return negative_frames

    def close(self):
        """Close the CVAT client connection."""
        self.cvat_client.logout()


def result_to_dict(result) -> Dict[str, Any]:
    """Convert result dataclass to dictionary for JSON serialization."""
    if isinstance(result, DetectionResult):
        return {
            "task_id": result.task_id,
            "task_name": result.task_name,
            "frame_number": result.frame_number,
            "frame_name": result.frame_name,
            "object_detected": result.object_detected,
            "num_objects": result.num_objects,
            "confidence_scores": result.confidence_scores,
            "prompt": result.prompt,
            "error": result.error
        }
    elif isinstance(result, TaskAnalysisResult):
        return {
            "task_id": result.task_id,
            "task_name": result.task_name,
            "total_frames": result.total_frames,
            "frames_with_object": result.frames_with_object,
            "frames_without_object": result.frames_without_object,
            "detection_rate": result.detection_rate,
            "frame_results": [result_to_dict(f) for f in result.frame_results]
        }
    elif isinstance(result, ProjectAnalysisResult):
        return {
            "project_id": result.project_id,
            "project_name": result.project_name,
            "prompt": result.prompt,
            "total_tasks": result.total_tasks,
            "total_frames": result.total_frames,
            "frames_with_object": result.frames_with_object,
            "overall_detection_rate": result.overall_detection_rate,
            "task_results": [result_to_dict(t) for t in result.task_results]
        }
    return result
