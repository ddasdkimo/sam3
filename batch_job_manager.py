"""
Background Job Manager for CVAT-SAM3 Batch Processing

Allows batch jobs to run in the background, even after the web page is closed.
Provides job status tracking and progress monitoring.
"""

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobProgress:
    """Progress information for a job."""
    current_step: int = 0
    total_steps: int = 0
    current_task: str = ""
    current_frame: int = 0
    task_frames: int = 0
    detected_count: int = 0
    processed_frames: int = 0
    total_frames: int = 0
    created_tasks: List[Dict] = field(default_factory=list)
    # For ETA calculation
    start_timestamp: float = 0.0  # Unix timestamp when processing started

    # Streaming mode progress
    batches_queued: int = 0      # Batches waiting to upload
    batches_uploaded: int = 0    # Batches successfully uploaded
    images_uploaded: int = 0     # Total images uploaded
    upload_errors: int = 0       # Upload error count
    current_phase: str = "scanning"  # "scanning", "uploading", "completed"

    @property
    def percentage(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return (self.processed_frames / self.total_frames) * 100

    @property
    def upload_percentage(self) -> float:
        """Upload progress percentage."""
        if self.detected_count == 0:
            return 0.0
        return (self.images_uploaded / self.detected_count) * 100

    def get_eta_seconds(self) -> Optional[float]:
        """Calculate estimated time remaining in seconds."""
        if self.processed_frames == 0 or self.start_timestamp == 0:
            return None
        if self.processed_frames >= self.total_frames:
            return 0.0

        elapsed = time.time() - self.start_timestamp
        if elapsed <= 0:
            return None

        frames_per_second = self.processed_frames / elapsed
        remaining_frames = self.total_frames - self.processed_frames
        return remaining_frames / frames_per_second if frames_per_second > 0 else None

    def get_eta_formatted(self) -> str:
        """Get formatted ETA string."""
        eta_seconds = self.get_eta_seconds()
        if eta_seconds is None:
            return "計算中..."
        if eta_seconds == 0:
            return "即將完成"

        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)
        seconds = int(eta_seconds % 60)

        if hours > 0:
            return f"{hours} 小時 {minutes} 分鐘"
        elif minutes > 0:
            return f"{minutes} 分鐘 {seconds} 秒"
        else:
            return f"{seconds} 秒"

    def get_processing_speed(self) -> Optional[float]:
        """Get frames per second processing speed."""
        if self.processed_frames == 0 or self.start_timestamp == 0:
            return None
        elapsed = time.time() - self.start_timestamp
        if elapsed <= 0:
            return None
        return self.processed_frames / elapsed


@dataclass
class BatchJob:
    """Represents a background batch processing job."""
    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Job parameters
    source_project_id: int = 0
    source_project_name: str = ""
    target_project_id: int = 0
    target_project_name: str = ""
    prompt: str = ""
    label_name: str = ""
    confidence_threshold: float = 0.6
    images_per_task: int = 500
    train_ratio: float = 70
    test_ratio: float = 20
    val_ratio: float = 10
    merge_regions: bool = False
    merge_kernel_size: int = 15
    merge_method: str = "closing"

    # Server URLs
    source_server_url: str = ""
    target_server_url: str = ""

    # Progress tracking
    progress: JobProgress = field(default_factory=JobProgress)

    # Results
    error_message: str = ""
    result_summary: str = ""
    log_messages: List[str] = field(default_factory=list)

    def add_log(self, message: str):
        """Add a log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_messages.append(f"[{timestamp}] {message}")
        # Keep only last 100 messages
        if len(self.log_messages) > 100:
            self.log_messages = self.log_messages[-100:]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "source_project_id": self.source_project_id,
            "source_project_name": self.source_project_name,
            "target_project_id": self.target_project_id,
            "target_project_name": self.target_project_name,
            "prompt": self.prompt,
            "label_name": self.label_name,
            "confidence_threshold": self.confidence_threshold,
            "images_per_task": self.images_per_task,
            "train_ratio": self.train_ratio,
            "test_ratio": self.test_ratio,
            "val_ratio": self.val_ratio,
            "merge_regions": self.merge_regions,
            "merge_kernel_size": self.merge_kernel_size,
            "merge_method": self.merge_method,
            "source_server_url": self.source_server_url,
            "target_server_url": self.target_server_url,
            "progress": {
                "current_step": self.progress.current_step,
                "total_steps": self.progress.total_steps,
                "current_task": self.progress.current_task,
                "current_frame": self.progress.current_frame,
                "task_frames": self.progress.task_frames,
                "detected_count": self.progress.detected_count,
                "processed_frames": self.progress.processed_frames,
                "total_frames": self.progress.total_frames,
                "percentage": self.progress.percentage,
                "created_tasks": self.progress.created_tasks,
                "start_timestamp": self.progress.start_timestamp,
                "batches_queued": self.progress.batches_queued,
                "batches_uploaded": self.progress.batches_uploaded,
                "images_uploaded": self.progress.images_uploaded,
                "upload_errors": self.progress.upload_errors,
                "current_phase": self.progress.current_phase,
            },
            "error_message": self.error_message,
            "result_summary": self.result_summary,
            "log_messages": self.log_messages[-20:],  # Last 20 messages
        }


class BatchJobManager:
    """
    Manages background batch processing jobs.

    Jobs run in separate threads and their status persists even if
    the web client disconnects.
    """

    def __init__(self, state_file: Optional[str] = None):
        self.jobs: Dict[str, BatchJob] = {}
        self.job_threads: Dict[str, threading.Thread] = {}
        self.cancel_flags: Dict[str, threading.Event] = {}
        self.lock = threading.Lock()

        # State persistence
        self.state_file = state_file or "/tmp/cvat_sam3_jobs.json"
        self._load_state()

    def _load_state(self):
        """Load job state from file (for recovery after restart)."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    # Only load completed/failed jobs for history
                    # Running jobs would need to be restarted
                    for job_data in data.get("jobs", []):
                        if job_data["status"] in ["completed", "failed", "cancelled"]:
                            job = self._dict_to_job(job_data)
                            self.jobs[job.job_id] = job
                logger.info(f"Loaded {len(self.jobs)} jobs from state file")
        except Exception as e:
            logger.warning(f"Could not load state file: {e}")

    def _save_state(self):
        """Save job state to file."""
        try:
            with self.lock:
                data = {
                    "jobs": [job.to_dict() for job in self.jobs.values()],
                    "saved_at": datetime.now().isoformat()
                }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save state file: {e}")

    def _dict_to_job(self, data: Dict) -> BatchJob:
        """Convert dictionary back to BatchJob."""
        job = BatchJob(
            job_id=data["job_id"],
            status=JobStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
        )
        if data.get("started_at"):
            job.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            job.completed_at = datetime.fromisoformat(data["completed_at"])

        job.source_project_id = data.get("source_project_id", 0)
        job.source_project_name = data.get("source_project_name", "")
        job.target_project_id = data.get("target_project_id", 0)
        job.target_project_name = data.get("target_project_name", "")
        job.prompt = data.get("prompt", "")
        job.label_name = data.get("label_name", "")
        job.confidence_threshold = data.get("confidence_threshold", 0.6)
        job.images_per_task = data.get("images_per_task", 500)
        job.train_ratio = data.get("train_ratio", 70)
        job.test_ratio = data.get("test_ratio", 20)
        job.val_ratio = data.get("val_ratio", 10)
        job.merge_regions = data.get("merge_regions", False)
        job.merge_kernel_size = data.get("merge_kernel_size", 15)
        job.merge_method = data.get("merge_method", "closing")
        job.source_server_url = data.get("source_server_url", "")
        job.target_server_url = data.get("target_server_url", "")
        job.error_message = data.get("error_message", "")
        job.result_summary = data.get("result_summary", "")
        job.log_messages = data.get("log_messages", [])

        progress_data = data.get("progress", {})
        job.progress = JobProgress(
            current_step=progress_data.get("current_step", 0),
            total_steps=progress_data.get("total_steps", 0),
            current_task=progress_data.get("current_task", ""),
            current_frame=progress_data.get("current_frame", 0),
            task_frames=progress_data.get("task_frames", 0),
            detected_count=progress_data.get("detected_count", 0),
            processed_frames=progress_data.get("processed_frames", 0),
            total_frames=progress_data.get("total_frames", 0),
            created_tasks=progress_data.get("created_tasks", []),
            start_timestamp=progress_data.get("start_timestamp", 0.0),
            batches_queued=progress_data.get("batches_queued", 0),
            batches_uploaded=progress_data.get("batches_uploaded", 0),
            images_uploaded=progress_data.get("images_uploaded", 0),
            upload_errors=progress_data.get("upload_errors", 0),
            current_phase=progress_data.get("current_phase", "scanning"),
        )

        return job

    def create_job(
        self,
        source_project_id: int,
        source_project_name: str,
        target_project_id: int,
        target_project_name: str,
        prompt: str,
        label_name: str,
        confidence_threshold: float = 0.6,
        images_per_task: int = 500,
        train_ratio: float = 70,
        test_ratio: float = 20,
        val_ratio: float = 10,
        merge_regions: bool = False,
        merge_kernel_size: int = 15,
        merge_method: str = "closing",
        source_server_url: str = "",
        target_server_url: str = "",
    ) -> BatchJob:
        """Create a new batch job."""
        job_id = str(uuid.uuid4())[:8]

        job = BatchJob(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            source_project_id=source_project_id,
            source_project_name=source_project_name,
            target_project_id=target_project_id,
            target_project_name=target_project_name,
            prompt=prompt,
            label_name=label_name,
            confidence_threshold=confidence_threshold,
            images_per_task=images_per_task,
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            merge_regions=merge_regions,
            merge_kernel_size=merge_kernel_size,
            merge_method=merge_method,
            source_server_url=source_server_url,
            target_server_url=target_server_url,
        )

        with self.lock:
            self.jobs[job_id] = job
            self.cancel_flags[job_id] = threading.Event()

        job.add_log(f"任務已建立: 從 '{source_project_name}' 轉換到 '{target_project_name}'")
        job.add_log(f"搜尋物件: {prompt}, 標籤: {label_name}")

        self._save_state()
        return job

    def start_job(self, job_id: str, job_function: Callable[[BatchJob, threading.Event], None]):
        """Start a job in a background thread."""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if job.status == JobStatus.RUNNING:
            raise ValueError(f"Job {job_id} is already running")

        cancel_flag = self.cancel_flags.get(job_id)
        if not cancel_flag:
            cancel_flag = threading.Event()
            self.cancel_flags[job_id] = cancel_flag

        def thread_wrapper():
            try:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now()
                job.add_log("任務開始執行")
                self._save_state()

                job_function(job, cancel_flag)

                if cancel_flag.is_set():
                    job.status = JobStatus.CANCELLED
                    job.add_log("任務已取消")
                else:
                    job.status = JobStatus.COMPLETED
                    job.add_log("任務執行完成")

            except Exception as e:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.add_log(f"任務失敗: {e}")
                logger.exception(f"Job {job_id} failed")
            finally:
                job.completed_at = datetime.now()
                self._save_state()

        thread = threading.Thread(target=thread_wrapper, daemon=True)
        self.job_threads[job_id] = thread
        thread.start()

        return job

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status != JobStatus.RUNNING:
            return False

        cancel_flag = self.cancel_flags.get(job_id)
        if cancel_flag:
            cancel_flag.set()
            job.add_log("收到取消請求，正在停止...")
            return True

        return False

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> List[BatchJob]:
        """Get all jobs sorted by creation time (newest first)."""
        with self.lock:
            return sorted(
                self.jobs.values(),
                key=lambda j: j.created_at,
                reverse=True
            )

    def get_running_jobs(self) -> List[BatchJob]:
        """Get all currently running jobs."""
        with self.lock:
            return [j for j in self.jobs.values() if j.status == JobStatus.RUNNING]

    def delete_job(self, job_id: str) -> bool:
        """Delete a completed/failed/cancelled job."""
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status == JobStatus.RUNNING:
            return False  # Cannot delete running job

        with self.lock:
            del self.jobs[job_id]
            if job_id in self.cancel_flags:
                del self.cancel_flags[job_id]
            if job_id in self.job_threads:
                del self.job_threads[job_id]

        self._save_state()
        return True

    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Remove jobs older than specified hours."""
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)

        with self.lock:
            to_delete = [
                job_id for job_id, job in self.jobs.items()
                if job.status != JobStatus.RUNNING
                and job.created_at.timestamp() < cutoff
            ]

            for job_id in to_delete:
                del self.jobs[job_id]
                if job_id in self.cancel_flags:
                    del self.cancel_flags[job_id]
                if job_id in self.job_threads:
                    del self.job_threads[job_id]

        if to_delete:
            self._save_state()
            logger.info(f"Cleaned up {len(to_delete)} old jobs")


# Global job manager instance
_job_manager: Optional[BatchJobManager] = None


def get_job_manager() -> BatchJobManager:
    """Get or create the global job manager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = BatchJobManager()
    return _job_manager
