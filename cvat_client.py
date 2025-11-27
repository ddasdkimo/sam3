"""
CVAT API Client for SAM3 Integration
Handles authentication, project/task retrieval, and image downloading from CVAT.
"""

import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CVATConfig:
    """CVAT server configuration."""
    server_url: str
    username: str
    password: str
    token: Optional[str] = None


class CVATClient:
    """Client for interacting with CVAT REST API."""

    def __init__(self, config: CVATConfig):
        self.config = config
        self.base_url = config.server_url.rstrip("/")
        self.session = requests.Session()
        self.token = config.token

        if not self.token:
            self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with CVAT and obtain API token."""
        url = f"{self.base_url}/api/auth/login"
        payload = {
            "username": self.config.username,
            "password": self.config.password
        }

        try:
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            self.token = data.get("key")
            logger.info(f"Successfully authenticated as {self.config.username}")
        except requests.RequestException as e:
            logger.error(f"Authentication failed: {e}")
            raise

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Token {self.token}",
            "Content-Type": "application/json"
        }

    def get_projects(self, page_size: int = 100) -> List[Dict[str, Any]]:
        """Get all projects from CVAT."""
        projects = []
        page = 1

        while True:
            url = f"{self.base_url}/api/projects?page={page}&page_size={page_size}"
            response = self.session.get(url, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
            data = response.json()

            projects.extend(data.get("results", []))

            if not data.get("next"):
                break
            page += 1

        logger.info(f"Retrieved {len(projects)} projects")
        return projects

    def get_project(self, project_id: int) -> Dict[str, Any]:
        """Get a specific project by ID."""
        url = f"{self.base_url}/api/projects/{project_id}"
        response = self.session.get(url, headers=self._get_headers(), timeout=30)
        response.raise_for_status()
        return response.json()

    def get_project_tasks(self, project_id: int, page_size: int = 100) -> List[Dict[str, Any]]:
        """Get all tasks for a specific project."""
        tasks = []
        page = 1

        while True:
            url = f"{self.base_url}/api/tasks?project_id={project_id}&page={page}&page_size={page_size}"
            response = self.session.get(url, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
            data = response.json()

            tasks.extend(data.get("results", []))

            if not data.get("next"):
                break
            page += 1

        logger.info(f"Retrieved {len(tasks)} tasks for project {project_id}")
        return tasks

    def get_task(self, task_id: int) -> Dict[str, Any]:
        """Get a specific task by ID."""
        url = f"{self.base_url}/api/tasks/{task_id}"
        response = self.session.get(url, headers=self._get_headers(), timeout=30)
        response.raise_for_status()
        return response.json()

    def get_task_data_meta(self, task_id: int) -> Dict[str, Any]:
        """Get metadata about task data (frames info)."""
        url = f"{self.base_url}/api/tasks/{task_id}/data/meta"
        response = self.session.get(url, headers=self._get_headers(), timeout=30)
        response.raise_for_status()
        return response.json()

    def get_task_frame(self, task_id: int, frame_number: int, quality: str = "original") -> Image.Image:
        """
        Download a specific frame from a task.

        Args:
            task_id: The task ID
            frame_number: Frame number (0-indexed)
            quality: 'original' or 'compressed'

        Returns:
            PIL Image object
        """
        url = f"{self.base_url}/api/tasks/{task_id}/data"
        params = {
            "type": "frame",
            "number": frame_number,
            "quality": quality
        }

        # Only use Authorization header for image download (no Content-Type)
        headers = {"Authorization": f"Token {self.token}"}

        response = self.session.get(
            url,
            headers=headers,
            params=params,
            timeout=60
        )
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content))
        return image

    def get_task_frames_info(self, task_id: int) -> List[Dict[str, Any]]:
        """Get information about all frames in a task."""
        meta = self.get_task_data_meta(task_id)
        frames = meta.get("frames", [])
        return frames

    def get_all_task_images(
        self,
        task_id: int,
        quality: str = "original",
        max_frames: Optional[int] = None
    ) -> List[tuple[int, Image.Image]]:
        """
        Download all images/frames from a task.

        Args:
            task_id: The task ID
            quality: 'original' or 'compressed'
            max_frames: Maximum number of frames to download (None for all)

        Returns:
            List of tuples (frame_number, PIL Image)
        """
        task = self.get_task(task_id)
        total_frames = task.get("size", 0)

        if max_frames:
            total_frames = min(total_frames, max_frames)

        images = []
        for frame_num in range(total_frames):
            try:
                image = self.get_task_frame(task_id, frame_num, quality)
                images.append((frame_num, image))
                logger.debug(f"Downloaded frame {frame_num + 1}/{total_frames} from task {task_id}")
            except Exception as e:
                logger.warning(f"Failed to download frame {frame_num} from task {task_id}: {e}")

        logger.info(f"Downloaded {len(images)} frames from task {task_id}")
        return images

    def get_labels(self, project_id: Optional[int] = None, task_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get labels for a project or task."""
        if project_id:
            url = f"{self.base_url}/api/labels?project_id={project_id}"
        elif task_id:
            url = f"{self.base_url}/api/labels?task_id={task_id}"
        else:
            raise ValueError("Either project_id or task_id must be provided")

        response = self.session.get(url, headers=self._get_headers(), timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])

    def create_task(
        self,
        name: str,
        project_id: int,
        labels: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Create a new task in a project.

        Args:
            name: Task name
            project_id: Project ID to add the task to
            labels: Optional list of label definitions (if not using project labels)

        Returns:
            Created task data
        """
        url = f"{self.base_url}/api/tasks"
        payload = {
            "name": name,
            "project_id": project_id,
        }
        if labels:
            payload["labels"] = labels

        response = self.session.post(url, headers=self._get_headers(), json=payload, timeout=30)
        response.raise_for_status()
        task = response.json()
        logger.info(f"Created task '{name}' with ID {task['id']} in project {project_id}")
        return task

    def upload_task_images(
        self,
        task_id: int,
        images: List[tuple[str, bytes]],
        image_quality: int = 70
    ) -> Dict[str, Any]:
        """
        Upload images to a task.

        Args:
            task_id: Task ID to upload images to
            images: List of tuples (filename, image_bytes)
            image_quality: Image quality for compression (0-100)

        Returns:
            Upload response data
        """
        url = f"{self.base_url}/api/tasks/{task_id}/data"

        files = []
        for filename, image_bytes in images:
            files.append(('client_files[0]', (filename, image_bytes, 'image/jpeg')))

        # For file upload, don't set Content-Type (let requests set multipart boundary)
        headers = {"Authorization": f"Token {self.token}"}

        data = {
            "image_quality": image_quality,
        }

        response = self.session.post(
            url,
            headers=headers,
            files=files,
            data=data,
            timeout=120
        )
        response.raise_for_status()
        logger.info(f"Uploaded {len(images)} images to task {task_id}")
        return response.json() if response.content else {}

    def upload_task_image_pil(
        self,
        task_id: int,
        image: Image.Image,
        filename: str = "image.jpg",
        image_quality: int = 70
    ) -> Dict[str, Any]:
        """
        Upload a PIL Image to a task.

        Args:
            task_id: Task ID
            image: PIL Image object
            filename: Filename for the image
            image_quality: JPEG quality

        Returns:
            Upload response data
        """
        # Convert PIL Image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=95)
        img_bytes = img_buffer.getvalue()

        return self.upload_task_images(task_id, [(filename, img_bytes)], image_quality)

    def get_job_id_for_task(self, task_id: int) -> int:
        """Get the first job ID for a task."""
        url = f"{self.base_url}/api/jobs?task_id={task_id}"
        response = self.session.get(url, headers=self._get_headers(), timeout=30)
        response.raise_for_status()
        data = response.json()
        jobs = data.get("results", [])
        if not jobs:
            raise ValueError(f"No jobs found for task {task_id}")
        return jobs[0]["id"]

    def upload_annotations(
        self,
        task_id: int,
        annotations: Dict[str, Any],
        format_name: str = "CVAT for images 1.1"
    ) -> None:
        """
        Upload annotations to a task.

        Args:
            task_id: Task ID
            annotations: Annotation data in CVAT format
            format_name: Annotation format
        """
        url = f"{self.base_url}/api/tasks/{task_id}/annotations"

        response = self.session.put(
            url,
            headers=self._get_headers(),
            json=annotations,
            timeout=60
        )
        response.raise_for_status()
        logger.info(f"Uploaded annotations to task {task_id}")

    def create_polygon_annotation(
        self,
        label_name: str,
        points: List[float],
        frame: int = 0,
        occluded: bool = False,
        z_order: int = 0
    ) -> Dict[str, Any]:
        """
        Create a polygon annotation object.

        Args:
            label_name: Name of the label
            points: List of x,y coordinates [x1,y1,x2,y2,...]
            frame: Frame number
            occluded: Whether the object is occluded
            z_order: Z-order for layering

        Returns:
            Polygon annotation dict
        """
        return {
            "type": "polygon",
            "occluded": occluded,
            "z_order": z_order,
            "points": points,
            "frame": frame,
            "label": label_name,
            "attributes": []
        }

    def merge_mask_regions(
        self,
        mask: np.ndarray,
        kernel_size: int = 15,
        method: str = "closing"
    ) -> np.ndarray:
        """
        使用形態學操作融合 mask 中分離的區域。

        Args:
            mask: Binary mask (numpy array)
            kernel_size: 核大小，越大融合越多分離的區域 (建議 5-30)
            method: 融合方法
                - "closing": 閉運算 (膨脹後腐蝕)，填補間隙但保持原始大小
                - "dilate": 只膨脹，區域會變大
                - "convex": 凸包，包圍所有區域

        Returns:
            處理後的 mask
        """
        import cv2
        import numpy as np

        if mask.ndim > 2:
            mask = mask.squeeze()

        mask_binary = (mask > 0.5).astype(np.uint8)

        if method == "closing":
            # 閉運算：先膨脹連接區域，再腐蝕恢復大小
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            merged = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)

        elif method == "dilate":
            # 純膨脹：區域會擴大
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            merged = cv2.dilate(mask_binary, kernel, iterations=1)

        elif method == "convex":
            # 凸包：找到所有區域的凸包
            contours, _ = cv2.findContours(
                mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                # 合併所有輪廓點
                all_points = np.vstack(contours)
                hull = cv2.convexHull(all_points)
                merged = np.zeros_like(mask_binary)
                cv2.fillPoly(merged, [hull], 1)
            else:
                merged = mask_binary

        else:
            merged = mask_binary

        return merged

    def mask_to_polygon(
        self,
        mask,
        tolerance: float = 2.0,
        merge_regions: bool = False,
        merge_kernel_size: int = 15,
        merge_method: str = "closing"
    ) -> List[List[float]]:
        """
        Convert a binary mask to polygon points.

        Args:
            mask: Binary mask (numpy array or tensor)
            tolerance: Approximation tolerance for polygon simplification
            merge_regions: 是否融合分離的區域
            merge_kernel_size: 融合核大小 (越大融合越多)
            merge_method: 融合方法 ("closing", "dilate", "convex")

        Returns:
            List of polygons, each polygon is a list of [x1,y1,x2,y2,...] points
        """
        import cv2
        import numpy as np

        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()

        if mask.ndim > 2:
            mask = mask.squeeze()

        # Ensure binary mask
        mask_binary = (mask > 0.5).astype(np.uint8)

        # 如果啟用融合，先處理 mask
        if merge_regions:
            mask_binary = self.merge_mask_regions(
                mask_binary,
                kernel_size=merge_kernel_size,
                method=merge_method
            )

        # Find contours
        contours, _ = cv2.findContours(
            mask_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []
        for contour in contours:
            # Simplify contour
            epsilon = tolerance
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Need at least 3 points for a polygon
            if len(approx) >= 3:
                # Flatten to [x1,y1,x2,y2,...]
                points = approx.flatten().tolist()
                polygons.append(points)

        return polygons

    def create_task_with_detection_results(
        self,
        project_id: int,
        task_name: str,
        image: Image.Image,
        masks,
        label_name: str,
        scores: Optional[List[float]] = None,
        min_polygon_points: int = 3
    ) -> Dict[str, Any]:
        """
        Create a new task with an image and upload SAM3 detection results as polygon annotations.

        Args:
            project_id: Project ID
            task_name: Name for the new task
            image: PIL Image
            masks: Detection masks from SAM3 (numpy array or tensor)
            label_name: Label name for the annotations
            scores: Optional confidence scores
            min_polygon_points: Minimum points required for a valid polygon

        Returns:
            Dict with task_id, job_id, and annotation counts
        """
        import numpy as np

        # Step 0: Get label_id from project labels
        labels = self.get_labels(project_id=project_id)
        label_id = None
        for label in labels:
            if label.get("name") == label_name:
                label_id = label.get("id")
                break

        if label_id is None:
            raise ValueError(f"Label '{label_name}' not found in project {project_id}")

        # Step 1: Create the task
        task = self.create_task(task_name, project_id)
        task_id = task["id"]

        # Step 2: Upload the image
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=95)
        img_bytes = img_buffer.getvalue()

        url = f"{self.base_url}/api/tasks/{task_id}/data"
        headers = {"Authorization": f"Token {self.token}"}

        files = [('client_files[0]', (f'{task_name}.jpg', img_bytes, 'image/jpeg'))]
        data = {"image_quality": 70}

        response = self.session.post(url, headers=headers, files=files, data=data, timeout=120)
        response.raise_for_status()

        # Wait for data to be processed
        import time
        for _ in range(30):
            task_info = self.get_task(task_id)
            if task_info.get("size", 0) > 0:
                break
            time.sleep(1)

        # Step 3: Convert masks to polygons
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if isinstance(masks, np.ndarray) and masks.ndim == 4:
            masks = masks.squeeze(1)

        shapes = []
        group_id_counter = 1  # CVAT group_id starts from 1
        for i, mask in enumerate(masks):
            polygons = self.mask_to_polygon(mask)
            # 同一個 mask 產生的多個 polygon 使用相同的 group_id
            # 這樣在 CVAT 中會被視為同一個物件的不同部分
            current_group_id = group_id_counter if len(polygons) > 1 else 0
            if len(polygons) > 1:
                group_id_counter += 1

            for polygon_points in polygons:
                if len(polygon_points) >= min_polygon_points * 2:  # x,y pairs
                    shape = {
                        "type": "polygon",
                        "occluded": False,
                        "z_order": 0,
                        "points": polygon_points,
                        "frame": 0,
                        "label_id": label_id,
                        "group": current_group_id,  # 設定 group_id
                        "attributes": []
                    }
                    shapes.append(shape)

        # Step 4: Upload annotations
        if shapes:
            annotations = {
                "version": 0,
                "tags": [],
                "shapes": shapes,
                "tracks": []
            }

            url = f"{self.base_url}/api/tasks/{task_id}/annotations"
            response = self.session.put(url, headers=self._get_headers(), json=annotations, timeout=60)
            response.raise_for_status()

        logger.info(f"Created task {task_id} with {len(shapes)} polygon annotations")

        return {
            "task_id": task_id,
            "task_name": task_name,
            "project_id": project_id,
            "num_annotations": len(shapes),
            "label_name": label_name
        }

    def logout(self) -> None:
        """Logout from CVAT."""
        url = f"{self.base_url}/api/auth/logout"
        try:
            self.session.post(url, headers=self._get_headers(), timeout=30)
            logger.info("Successfully logged out from CVAT")
        except requests.RequestException as e:
            logger.warning(f"Logout request failed: {e}")
        finally:
            self.session.close()
