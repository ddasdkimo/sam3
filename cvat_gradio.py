"""
CVAT-SAM3 Gradio Web Interface
Interactive UI for browsing CVAT projects and detecting objects with SAM3.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from cvat_client import CVATClient, CVATConfig
from batch_job_manager import BatchJobManager, BatchJob, JobStatus, get_job_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
_cvat_client: Optional[CVATClient] = None
_sam3_model = None
_sam3_processor = None

# Cache for projects, tasks, frames
_projects_cache: List[Dict] = []
_tasks_cache: Dict[int, List[Dict]] = {}
_frames_cache: Dict[int, Dict] = {}


def get_cvat_client() -> CVATClient:
    """Get or create CVAT client."""
    global _cvat_client
    if _cvat_client is None:
        config = CVATConfig(
            server_url=os.getenv("CVAT_SERVER_URL", "http://192.168.50.15:8080"),
            username=os.getenv("CVAT_USERNAME", "david"),
            password=os.getenv("CVAT_PASSWORD", "a123321a"),
        )
        _cvat_client = CVATClient(config)
    return _cvat_client


def get_sam3_model():
    """Lazy load SAM3 model."""
    global _sam3_model, _sam3_processor
    if _sam3_model is None:
        logger.info("Loading SAM3 model...")
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _sam3_model = build_sam3_image_model(
            device=device,
            eval_mode=True,
            load_from_HF=True,
        )
        _sam3_processor = Sam3Processor(_sam3_model, device=device)
        logger.info(f"SAM3 model loaded on {device}")
    return _sam3_model, _sam3_processor


def load_projects() -> List[Tuple[str, int]]:
    """Load all projects from CVAT."""
    global _projects_cache
    try:
        client = get_cvat_client()
        _projects_cache = client.get_projects()
        choices = [(f"{p['name']} (ID: {p['id']})", p['id']) for p in _projects_cache]
        return choices
    except Exception as e:
        logger.error(f"Error loading projects: {e}")
        return []


def load_tasks(project_id: int) -> List[Tuple[str, int]]:
    """Load tasks for a project."""
    global _tasks_cache
    if not project_id:
        return []
    try:
        client = get_cvat_client()
        tasks = client.get_project_tasks(project_id)
        _tasks_cache[project_id] = tasks
        choices = [(f"{t['name']} ({t.get('size', 0)} frames)", t['id']) for t in tasks]
        return choices
    except Exception as e:
        logger.error(f"Error loading tasks: {e}")
        return []


def load_frames(task_id: int) -> Tuple[List[Tuple[str, int]], Dict]:
    """Load frame info for a task."""
    global _frames_cache
    if not task_id:
        return [], {}
    try:
        client = get_cvat_client()
        task = client.get_task(task_id)
        meta = client.get_task_data_meta(task_id)

        total_frames = task.get("size", 0)
        frames_info = meta.get("frames", [])

        _frames_cache[task_id] = {
            "total": total_frames,
            "info": frames_info
        }

        choices = []
        for i in range(total_frames):
            if i < len(frames_info):
                name = frames_info[i].get("name", f"Frame {i}")
                choices.append((f"{i}: {name}", i))
            else:
                choices.append((f"Frame {i}", i))

        return choices, {"total": total_frames}
    except Exception as e:
        logger.error(f"Error loading frames: {e}")
        return [], {}


def get_frame_image(task_id: int, frame_number: int) -> Optional[Image.Image]:
    """Download a frame from CVAT."""
    if not task_id or frame_number is None:
        return None
    try:
        client = get_cvat_client()
        image = client.get_task_frame(task_id, frame_number, "compressed")
        return image
    except Exception as e:
        logger.error(f"Error getting frame: {e}")
        return None


def overlay_masks_on_image(
    image: np.ndarray,
    masks: np.ndarray,
    scores: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None
) -> np.ndarray:
    """Overlay segmentation masks on the image with labels."""
    if len(masks) == 0:
        return image

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    result = image.copy().astype(np.float32)

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
    ]

    alpha = 0.5
    for i, mask in enumerate(masks):
        if mask.ndim > 2:
            mask = mask.squeeze()

        color = colors[i % len(colors)]
        mask_bool = mask > 0.5

        for c in range(3):
            result[:, :, c] = np.where(
                mask_bool,
                result[:, :, c] * (1 - alpha) + color[c] * alpha,
                result[:, :, c]
            )

        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color, 2)

        # Add label
        if contours and len(contours) > 0:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                score_text = f"{scores[i]*100:.1f}%" if scores is not None else ""
                label_text = f"#{i+1} {score_text}"

                cv2.putText(
                    result, label_text, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                cv2.putText(
                    result, label_text, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1
                )

    return result.astype(np.uint8)


def detect_objects(
    image: Image.Image,
    prompt: str,
    confidence_threshold: float
) -> Tuple[np.ndarray, Dict]:
    """Detect objects in image using SAM3."""
    if image is None or not prompt:
        return None, {}

    try:
        model, processor = get_sam3_model()
        processor.confidence_threshold = confidence_threshold

        state = processor.set_image(image)
        output = processor.set_text_prompt(prompt=prompt, state=state)

        masks = output.get("masks", None)
        scores = output.get("scores", None)
        boxes = output.get("boxes", None)

        if masks is None or len(masks) == 0:
            return np.array(image), {
                "detected": False,
                "num_objects": 0,
                "message": f"未找到 '{prompt}'"
            }

        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        result_image = overlay_masks_on_image(np.array(image), masks, scores)

        return result_image, {
            "detected": True,
            "num_objects": len(masks),
            "scores": scores.tolist() if scores is not None else [],
            "masks": masks,  # Keep raw masks for CVAT upload
            "message": f"找到 {len(masks)} 個 '{prompt}'"
        }

    except Exception as e:
        logger.error(f"Detection error: {e}")
        return np.array(image), {"error": str(e)}


# Store last detection result for CVAT upload
_last_detection = {
    "image": None,
    "masks": None,
    "scores": None,
    "prompt": None
}


def detect_and_store(task_id, frame_number, prompt, confidence):
    """Detect objects and store results for CVAT upload."""
    import time
    global _last_detection

    if task_id is None or frame_number is None or not prompt:
        return None, "請選擇任務、幀並輸入搜尋物件", gr.update(interactive=False)

    # Time image download
    t0 = time.time()
    image = get_frame_image(task_id, frame_number)
    download_time = time.time() - t0

    if image is None:
        return None, "無法載入圖片", gr.update(interactive=False)

    # Time detection
    t1 = time.time()
    result_image, info = detect_objects(image, prompt, confidence)
    detection_time = time.time() - t1

    total_time = download_time + detection_time

    if "error" in info:
        _last_detection = {"image": None, "masks": None, "scores": None, "prompt": None}
        return result_image, f"錯誤: {info['error']}", gr.update(interactive=False)

    # Store for CVAT upload
    _last_detection = {
        "image": image,
        "masks": info.get("masks"),
        "scores": info.get("scores"),
        "prompt": prompt
    }

    result_text = f"""
**搜尋物件**: {prompt}
**結果**: {info['message']}

**⏱️ 處理時間**:
- 圖片下載: {download_time:.2f} 秒
- SAM3 辨識: {detection_time:.2f} 秒
- **總計: {total_time:.2f} 秒**
"""
    if info.get("detected") and info.get("scores"):
        result_text += "\n**信心度**:\n"
        for i, score in enumerate(info["scores"]):
            result_text += f"  - 物件 {i+1}: {score*100:.1f}%\n"

    # Enable upload button if detection succeeded
    can_upload = info.get("detected", False) and info.get("masks") is not None
    return result_image, result_text, gr.update(interactive=can_upload)


def load_project_labels(project_id):
    """Load labels for a project."""
    if not project_id:
        return gr.update(choices=[], value=None)

    try:
        client = get_cvat_client()
        labels = client.get_labels(project_id=project_id)
        choices = [(lbl["name"], lbl["name"]) for lbl in labels]
        return gr.update(choices=choices, value=choices[0][0] if choices else None)
    except Exception as e:
        logger.error(f"Error loading labels: {e}")
        return gr.update(choices=[], value=None)


def upload_to_cvat(target_project_id, task_name, label_name, progress=gr.Progress()):
    """Upload detection results to CVAT as a new task with polygon annotations."""
    global _last_detection

    if not target_project_id:
        return "請選擇目標專案"

    if not task_name:
        return "請輸入任務名稱"

    if not label_name:
        return "請選擇標籤"

    if _last_detection["image"] is None or _last_detection["masks"] is None:
        return "請先執行檢測，確保有檢測結果"

    try:
        progress(0.1, desc="建立任務...")
        client = get_cvat_client()

        progress(0.5, desc="上傳圖片和標註...")
        result = client.create_task_with_detection_results(
            project_id=target_project_id,
            task_name=task_name,
            image=_last_detection["image"],
            masks=_last_detection["masks"],
            label_name=label_name,
            scores=_last_detection["scores"]
        )

        progress(1.0, desc="完成!")

        return f"""
## ✅ 上傳成功!

- **任務 ID**: {result['task_id']}
- **任務名稱**: {result['task_name']}
- **專案 ID**: {result['project_id']}
- **標籤**: {result['label_name']}
- **Polygon 標註數**: {result['num_annotations']}

[在 CVAT 中查看任務](http://192.168.50.15:8080/tasks/{result['task_id']})
"""

    except Exception as e:
        logger.error(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ 上傳失敗: {str(e)}"


def analyze_task_frames(
    task_id: int,
    prompt: str,
    confidence_threshold: float,
    max_frames: int,
    progress=gr.Progress()
) -> Tuple[str, List]:
    """Analyze multiple frames in a task."""
    if not task_id or not prompt:
        return "請選擇任務並輸入搜尋物件", []

    try:
        client = get_cvat_client()
        task = client.get_task(task_id)
        total_frames = min(task.get("size", 0), max_frames)

        results = []
        frames_with_object = 0

        progress(0, desc="開始分析...")

        for i in range(total_frames):
            progress((i + 1) / total_frames, desc=f"分析 Frame {i + 1}/{total_frames}")

            image = client.get_task_frame(task_id, i, "compressed")
            result_image, detection_info = detect_objects(image, prompt, confidence_threshold)

            if detection_info.get("detected", False):
                frames_with_object += 1
                results.append((result_image, f"Frame {i}: {detection_info['message']}"))

        summary = f"""
## 分析結果

- **任務**: {task.get('name', task_id)}
- **搜尋物件**: {prompt}
- **分析幀數**: {total_frames}
- **包含物件的幀**: {frames_with_object}
- **檢測率**: {frames_with_object/total_frames*100:.1f}%
"""
        return summary, results

    except Exception as e:
        logger.error(f"Task analysis error: {e}")
        return f"錯誤: {str(e)}", []


# Global state for batch processing
_batch_stop_flag = False


def batch_scan_project(
    source_project_id: int,
    target_project_id: int,
    prompt: str,
    label_name: str,
    confidence_threshold: float,
    images_per_task: int,
    train_ratio: float = 70,
    test_ratio: float = 20,
    val_ratio: float = 10,
    merge_regions: bool = False,
    merge_kernel_size: int = 15,
    merge_method: str = "closing",
    progress=gr.Progress()
):
    """
    Scan all tasks/frames in a project, detect objects, and transfer to target project.
    Creates new tasks with max images_per_task images each.
    Splits into Train/Test/Validation sets based on ratios.
    """
    global _batch_stop_flag
    _batch_stop_flag = False

    if not source_project_id:
        yield "❌ 請選擇來源專案", "", []
        return

    if not target_project_id:
        yield "❌ 請選擇目標專案", "", []
        return

    if not prompt:
        yield "❌ 請輸入搜尋物件", "", []
        return

    if not label_name:
        yield "❌ 請選擇標籤", "", []
        return

    try:
        import time
        client = get_cvat_client()

        # Get source project info
        source_project = client.get_project(source_project_id)
        source_tasks = client.get_project_tasks(source_project_id)

        # Calculate total frames
        total_frames = sum(t.get("size", 0) for t in source_tasks)
        total_tasks = len(source_tasks)

        yield f"📊 開始掃描專案: {source_project['name']}\n總任務數: {total_tasks}\n總幀數: {total_frames}", "", []

        # Collect detected images and masks
        detected_images = []  # List of (image, masks, scores, source_info)
        processed_frames = 0
        detected_count = 0

        progress(0, desc="掃描中...")

        for task_idx, task in enumerate(source_tasks):
            if _batch_stop_flag:
                yield f"⏹️ 已停止掃描\n已處理: {processed_frames}/{total_frames} 幀", "", []
                return

            task_id = task["id"]
            task_name = task["name"]
            task_size = task.get("size", 0)

            for frame_num in range(task_size):
                if _batch_stop_flag:
                    yield f"⏹️ 已停止掃描\n已處理: {processed_frames}/{total_frames} 幀", "", []
                    return

                processed_frames += 1
                progress_pct = processed_frames / total_frames
                progress(progress_pct, desc=f"掃描 {task_name} - Frame {frame_num+1}/{task_size}")

                try:
                    # Download frame
                    image = client.get_task_frame(task_id, frame_num, "compressed")

                    # Detect objects
                    _, detection_info = detect_objects(image, prompt, confidence_threshold)

                    if detection_info.get("detected", False) and detection_info.get("masks") is not None:
                        detected_count += 1
                        detected_images.append({
                            "image": image,
                            "masks": detection_info["masks"],
                            "scores": detection_info.get("scores"),
                            "source_task": task_name,
                            "source_frame": frame_num
                        })

                        # Update status every detection
                        status = f"""
🔍 **掃描進度**: {processed_frames}/{total_frames} 幀 ({progress_pct*100:.1f}%)
📁 **當前任務**: {task_name} (Frame {frame_num+1}/{task_size})
✅ **已檢測到**: {detected_count} 張包含 '{prompt}' 的圖片
"""
                        yield status, "", []

                except Exception as e:
                    logger.warning(f"Error processing {task_name} frame {frame_num}: {e}")
                    continue

        # Now create tasks in target project (1000 images per task)
        if not detected_images:
            yield f"""
## ✅ 掃描完成

- 掃描總幀數: {total_frames}
- 檢測到物件: 0 張
- 未找到包含 '{prompt}' 的圖片
""", "", []
            return

        yield f"📤 開始上傳 {detected_count} 張圖片到目標專案...", "", []

        # Shuffle and split into Train/Test/Validation sets
        import random
        random.shuffle(detected_images)

        total_images = len(detected_images)
        # Normalize ratios
        ratio_sum = train_ratio + test_ratio + val_ratio
        if ratio_sum <= 0:
            ratio_sum = 100
            train_ratio, test_ratio, val_ratio = 70, 20, 10

        train_count = int(total_images * train_ratio / ratio_sum)
        test_count = int(total_images * test_ratio / ratio_sum)
        val_count = total_images - train_count - test_count  # 剩餘給 validation

        # Split images into sets
        dataset_splits = {}
        if train_ratio > 0 and train_count > 0:
            dataset_splits["train"] = detected_images[:train_count]
        if test_ratio > 0 and test_count > 0:
            dataset_splits["test"] = detected_images[train_count:train_count + test_count]
        if val_ratio > 0 and val_count > 0:
            dataset_splits["val"] = detected_images[train_count + test_count:]

        yield f"""📊 資料集分割:
- Train: {train_count} 張 ({train_ratio:.0f}%)
- Test: {test_count} 張 ({test_ratio:.0f}%)
- Validation: {val_count} 張 ({val_ratio:.0f}%)
""", "", []

        created_tasks = []
        total_annotations = 0
        total_chunks = 0

        # Calculate total chunks for progress
        for split_name, split_images in dataset_splits.items():
            total_chunks += (len(split_images) + images_per_task - 1) // images_per_task

        current_chunk = 0

        # Process each split
        for split_name, split_images in dataset_splits.items():
            if _batch_stop_flag:
                break

            # Split into chunks of images_per_task
            chunks = [split_images[i:i + images_per_task]
                      for i in range(0, len(split_images), images_per_task)]

            for chunk_idx, chunk in enumerate(chunks):
                if _batch_stop_flag:
                    break

                current_chunk += 1
                task_num = chunk_idx + 1
                # Include source project name and split in task name
                source_project_name = source_project['name'].replace(' ', '_')[:25]
                task_name = f"{source_project_name}_{prompt}_{split_name}_{task_num:03d}"

                progress(current_chunk / total_chunks, desc=f"建立 {split_name} 任務 {task_num}/{len(chunks)}")

                yield f"📦 建立 {split_name} 任務 {task_num}/{len(chunks)}: {task_name} ({len(chunk)} 張圖片)", "", []

                try:
                    # Prepare image files and create ZIP archive
                    import io
                    import zipfile

                    files = []
                    # Use sequential numbering to ensure CVAT's alphabetical sort matches our order
                    # Store original info for annotation creation
                    file_metadata = []
                    for img_idx, item in enumerate(chunk):
                        img_buffer = io.BytesIO()
                        item["image"].save(img_buffer, format='JPEG', quality=95)
                        img_bytes = img_buffer.getvalue()
                        # Use zero-padded sequential number as filename prefix to ensure correct sort order
                        filename = f"{img_idx:06d}_{item['source_task']}_frame{item['source_frame']:06d}.jpg"
                        files.append((filename, img_bytes))
                        file_metadata.append({
                            "filename": filename,
                            "item": item,
                            "expected_frame": img_idx  # This should match CVAT frame number after sort
                        })

                    # Create ZIP archive in memory
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for filename, img_bytes in files:
                            zf.writestr(filename, img_bytes)
                    zip_buffer.seek(0)
                    zip_bytes = zip_buffer.getvalue()
                    zip_size_mb = len(zip_bytes) / (1024 * 1024)
                    logger.info(f"Created ZIP archive: {len(files)} images, {zip_size_mb:.2f} MB")

                    # Retry with increasing timeout (includes task creation)
                    max_retries = 3
                    base_timeout = 600  # 10 minutes
                    upload_success = False
                    new_task_id = None

                    for retry in range(max_retries):
                        try:
                            # Delete previous failed task if exists
                            if new_task_id is not None:
                                try:
                                    logger.info(f"Deleting failed task {new_task_id}")
                                    delete_url = f"{client.base_url}/api/tasks/{new_task_id}"
                                    client.session.delete(delete_url, headers=client._get_headers(), timeout=30)
                                except Exception as del_err:
                                    logger.warning(f"Failed to delete task {new_task_id}: {del_err}")

                            # Create new task
                            retry_task_name = task_name if retry == 0 else f"{task_name}_retry{retry}"
                            new_task = client.create_task(retry_task_name, target_project_id)
                            new_task_id = new_task["id"]
                            logger.info(f"Created task {new_task_id}: {retry_task_name}")

                            # Upload ZIP archive (single file upload is more reliable)
                            url = f"{client.base_url}/api/tasks/{new_task_id}/data"
                            headers = {"Authorization": f"Token {client.token}"}

                            zip_filename = f"{task_name}.zip"
                            multipart_files = [
                                ('client_files[0]', (zip_filename, zip_bytes, 'application/zip'))
                            ]

                            data = {"image_quality": 70}

                            current_timeout = base_timeout * (retry + 1)  # 600, 1200, 1800 seconds
                            logger.info(f"Upload attempt {retry + 1}/{max_retries} with timeout {current_timeout}s, ZIP size: {zip_size_mb:.2f} MB")

                            response = client.session.post(
                                url, headers=headers, files=multipart_files,
                                data=data, timeout=current_timeout
                            )
                            response.raise_for_status()
                            upload_success = True
                            logger.info(f"Uploaded ZIP with {len(files)} images to task {new_task_id}")
                            break

                        except Exception as upload_error:
                            logger.warning(f"Upload attempt {retry + 1} failed: {upload_error}")
                            if retry < max_retries - 1:
                                logger.info(f"Retrying with longer timeout...")
                                time.sleep(5)  # Wait before retry
                            else:
                                # Clean up failed task on final failure
                                if new_task_id is not None:
                                    try:
                                        delete_url = f"{client.base_url}/api/tasks/{new_task_id}"
                                        client.session.delete(delete_url, headers=client._get_headers(), timeout=30)
                                        logger.info(f"Cleaned up failed task {new_task_id}")
                                    except:
                                        pass
                                raise upload_error

                    if not upload_success:
                        raise Exception(f"Failed to upload images after {max_retries} attempts")

                    # Wait for task to be ready (up to 5 minutes)
                    for _ in range(300):
                        task_info = client.get_task(new_task_id)
                        if task_info.get("size", 0) >= len(chunk):
                            break
                        time.sleep(1)

                    # Get frame metadata to build filename -> frame_number mapping
                    meta = client.get_task_data_meta(new_task_id)
                    frames_info = meta.get("frames", [])

                    # Build filename to frame number mapping
                    # CVAT may store full path or just filename, so we map both
                    filename_to_frame = {}
                    for frame_num, frame_info in enumerate(frames_info):
                        fname = frame_info.get("name", "")
                        # Map both full path and basename
                        filename_to_frame[fname] = frame_num
                        # Also map just the basename (in case CVAT stores with path)
                        import os
                        basename = os.path.basename(fname)
                        if basename != fname:
                            filename_to_frame[basename] = frame_num

                    logger.info(f"Frame mapping for task {new_task_id}: {len(frames_info)} frames")
                    if frames_info:
                        logger.info(f"Sample frame names: {[f.get('name', '') for f in frames_info[:3]]}")

                    # Get label_id
                    labels = client.get_labels(project_id=target_project_id)
                    label_id = None
                    for label in labels:
                        if label.get("name") == label_name:
                            label_id = label.get("id")
                            break

                    if label_id is None:
                        raise ValueError(f"Label '{label_name}' not found")

                    # Create annotations for each frame using correct frame mapping
                    shapes = []
                    mapping_misses = 0
                    group_id_counter = 1  # CVAT group_id starts from 1

                    for meta in file_metadata:
                        filename = meta["filename"]
                        item = meta["item"]
                        expected_frame = meta["expected_frame"]

                        # Find the actual frame number in CVAT
                        frame_num = filename_to_frame.get(filename)
                        if frame_num is None:
                            # Fallback: use expected frame number based on our sequential naming
                            mapping_misses += 1
                            logger.warning(f"Frame mapping miss for {filename}, using expected index {expected_frame}")
                            frame_num = expected_frame

                        masks = item["masks"]
                        if hasattr(masks, 'cpu'):
                            masks = masks.cpu().numpy()
                        if masks.ndim == 4:
                            masks = masks.squeeze(1)

                        for mask in masks:
                            polygons = client.mask_to_polygon(
                                mask,
                                merge_regions=merge_regions,
                                merge_kernel_size=merge_kernel_size,
                                merge_method=merge_method
                            )
                            # 同一個 mask 產生的多個 polygon 使用相同的 group_id
                            # 這樣在 CVAT 中會被視為同一個物件的不同部分
                            current_group_id = group_id_counter if len(polygons) > 1 else 0
                            if len(polygons) > 1:
                                group_id_counter += 1

                            for polygon_points in polygons:
                                if len(polygon_points) >= 6:  # At least 3 points
                                    shapes.append({
                                        "type": "polygon",
                                        "occluded": False,
                                        "z_order": 0,
                                        "points": polygon_points,
                                        "frame": frame_num,
                                        "label_id": label_id,
                                        "group": current_group_id,  # 設定 group_id
                                        "attributes": []
                                    })

                    # Log mapping summary
                    if mapping_misses > 0:
                        logger.warning(f"Task {new_task_id}: {mapping_misses}/{len(file_metadata)} frame mappings missed (using fallback)")
                    else:
                        logger.info(f"Task {new_task_id}: All {len(file_metadata)} frame mappings successful")

                    # Upload annotations
                    if shapes:
                        annotations = {
                            "version": 0,
                            "tags": [],
                            "shapes": shapes,
                            "tracks": []
                        }
                        url = f"{client.base_url}/api/tasks/{new_task_id}/annotations"
                        response = client.session.put(url, headers=client._get_headers(), json=annotations, timeout=60)
                        response.raise_for_status()
                        logger.info(f"Uploaded {len(shapes)} annotations to task {new_task_id}")

                    total_annotations += len(shapes)
                    created_tasks.append({
                        "task_id": new_task_id,
                        "task_name": task_name,
                        "split": split_name,
                        "images": len(chunk),
                        "annotations": len(shapes)
                    })

                except Exception as e:
                    logger.error(f"Error creating task {task_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Final summary with split info
        train_tasks = [t for t in created_tasks if t.get('split') == 'train']
        test_tasks = [t for t in created_tasks if t.get('split') == 'test']
        val_tasks = [t for t in created_tasks if t.get('split') == 'val']

        summary = f"""
## ✅ 批次處理完成!

### 掃描結果
- **來源專案**: {source_project['name']}
- **掃描任務數**: {total_tasks}
- **掃描總幀數**: {total_frames}
- **檢測到物件**: {detected_count} 張

### 資料集分割
- **Train**: {sum(t['images'] for t in train_tasks)} 張 ({len(train_tasks)} 個任務)
- **Test**: {sum(t['images'] for t in test_tasks)} 張 ({len(test_tasks)} 個任務)
- **Validation**: {sum(t['images'] for t in val_tasks)} 張 ({len(val_tasks)} 個任務)

### 輸出結果
- **建立任務數**: {len(created_tasks)}
- **總標註數**: {total_annotations}
- **目標專案 ID**: {target_project_id}

### 建立的任務:
"""
        for split_name in ['train', 'test', 'val']:
            split_tasks = [t for t in created_tasks if t.get('split') == split_name]
            if split_tasks:
                summary += f"\n**{split_name.upper()}:**\n"
                for t in split_tasks:
                    summary += f"- [{t['task_name']}](http://192.168.50.15:8080/tasks/{t['task_id']}) - {t['images']} 張圖片, {t['annotations']} 個標註\n"

        log_text = "\n".join([f"Task {t['task_id']}: {t['task_name']}" for t in created_tasks])

        yield summary, log_text, []

    except Exception as e:
        logger.error(f"Batch scan error: {e}")
        import traceback
        traceback.print_exc()
        yield f"❌ 錯誤: {str(e)}", "", []


def stop_batch_scan():
    """Stop the batch scanning process."""
    global _batch_stop_flag
    _batch_stop_flag = True
    return "⏹️ 正在停止..."


def run_batch_job_background(job: BatchJob, cancel_flag):
    """
    Execute batch scan in background thread.
    This function runs independently of the web UI.
    """
    import io
    import os
    import random
    import time
    import zipfile

    try:
        client = get_cvat_client()

        # Get source project info
        source_project = client.get_project(job.source_project_id)
        source_tasks = client.get_project_tasks(job.source_project_id)

        # Calculate total frames
        total_frames = sum(t.get("size", 0) for t in source_tasks)
        total_tasks = len(source_tasks)

        job.progress.total_frames = total_frames
        job.progress.total_steps = total_tasks
        job.add_log(f"來源專案: {source_project['name']}, 任務數: {total_tasks}, 總幀數: {total_frames}")

        # Collect detected images
        detected_images = []
        processed_frames = 0

        for task_idx, task in enumerate(source_tasks):
            if cancel_flag.is_set():
                job.add_log("收到取消請求，停止掃描")
                return

            task_id = task["id"]
            task_name = task["name"]
            task_size = task.get("size", 0)

            job.progress.current_step = task_idx + 1
            job.progress.current_task = task_name
            job.progress.task_frames = task_size
            job.add_log(f"掃描任務 {task_idx + 1}/{total_tasks}: {task_name}")

            for frame_num in range(task_size):
                if cancel_flag.is_set():
                    return

                processed_frames += 1
                job.progress.processed_frames = processed_frames
                job.progress.current_frame = frame_num + 1

                try:
                    image = client.get_task_frame(task_id, frame_num, "compressed")
                    _, detection_info = detect_objects(image, job.prompt, job.confidence_threshold)

                    if detection_info.get("detected", False) and detection_info.get("masks") is not None:
                        job.progress.detected_count += 1
                        detected_images.append({
                            "image": image,
                            "masks": detection_info["masks"],
                            "scores": detection_info.get("scores"),
                            "source_task": task_name,
                            "source_frame": frame_num
                        })

                except Exception as e:
                    logger.warning(f"Error processing {task_name} frame {frame_num}: {e}")
                    continue

        job.add_log(f"掃描完成: 處理 {processed_frames} 幀, 檢測到 {len(detected_images)} 張包含物件的圖片")

        if not detected_images:
            job.result_summary = f"掃描完成，未找到包含 '{job.prompt}' 的圖片"
            return

        # Shuffle and split into Train/Test/Validation
        random.shuffle(detected_images)
        total_images = len(detected_images)

        ratio_sum = job.train_ratio + job.test_ratio + job.val_ratio
        if ratio_sum <= 0:
            ratio_sum = 100
            job.train_ratio, job.test_ratio, job.val_ratio = 70, 20, 10

        train_count = int(total_images * job.train_ratio / ratio_sum)
        test_count = int(total_images * job.test_ratio / ratio_sum)
        val_count = total_images - train_count - test_count

        dataset_splits = {}
        if job.train_ratio > 0 and train_count > 0:
            dataset_splits["train"] = detected_images[:train_count]
        if job.test_ratio > 0 and test_count > 0:
            dataset_splits["test"] = detected_images[train_count:train_count + test_count]
        if job.val_ratio > 0 and val_count > 0:
            dataset_splits["val"] = detected_images[train_count + test_count:]

        job.add_log(f"資料集分割: Train={train_count}, Test={test_count}, Val={val_count}")

        # Create tasks in target project
        created_tasks = []
        total_annotations = 0

        for split_name, split_images in dataset_splits.items():
            if cancel_flag.is_set():
                break

            chunks = [split_images[i:i + job.images_per_task]
                      for i in range(0, len(split_images), job.images_per_task)]

            for chunk_idx, chunk in enumerate(chunks):
                if cancel_flag.is_set():
                    break

                task_num = chunk_idx + 1
                source_project_name = source_project['name'].replace(' ', '_')[:25]
                task_name = f"{source_project_name}_{job.prompt}_{split_name}_{task_num:03d}"

                job.add_log(f"建立 {split_name} 任務 {task_num}/{len(chunks)}: {task_name}")

                try:
                    # Prepare ZIP archive
                    files = []
                    file_metadata = []
                    for img_idx, item in enumerate(chunk):
                        img_buffer = io.BytesIO()
                        item["image"].save(img_buffer, format='JPEG', quality=95)
                        img_bytes = img_buffer.getvalue()
                        filename = f"{img_idx:06d}_{item['source_task']}_frame{item['source_frame']:06d}.jpg"
                        files.append((filename, img_bytes))
                        file_metadata.append({
                            "filename": filename,
                            "item": item,
                            "expected_frame": img_idx
                        })

                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for filename, img_bytes in files:
                            zf.writestr(filename, img_bytes)
                    zip_buffer.seek(0)
                    zip_bytes = zip_buffer.getvalue()

                    # Create task with retry
                    max_retries = 3
                    base_timeout = 600
                    new_task_id = None

                    for retry in range(max_retries):
                        try:
                            if new_task_id is not None:
                                try:
                                    delete_url = f"{client.base_url}/api/tasks/{new_task_id}"
                                    client.session.delete(delete_url, headers=client._get_headers(), timeout=30)
                                except:
                                    pass

                            retry_task_name = task_name if retry == 0 else f"{task_name}_retry{retry}"
                            new_task = client.create_task(retry_task_name, job.target_project_id)
                            new_task_id = new_task["id"]

                            url = f"{client.base_url}/api/tasks/{new_task_id}/data"
                            headers = {"Authorization": f"Token {client.token}"}
                            zip_filename = f"{task_name}.zip"
                            multipart_files = [('client_files[0]', (zip_filename, zip_bytes, 'application/zip'))]
                            data = {"image_quality": 70}

                            current_timeout = base_timeout * (retry + 1)
                            response = client.session.post(
                                url, headers=headers, files=multipart_files,
                                data=data, timeout=current_timeout
                            )
                            response.raise_for_status()
                            break

                        except Exception as upload_error:
                            if retry < max_retries - 1:
                                time.sleep(5)
                            else:
                                raise upload_error

                    # Wait for task to be ready
                    for _ in range(300):
                        task_info = client.get_task(new_task_id)
                        if task_info.get("size", 0) >= len(chunk):
                            break
                        time.sleep(1)

                    # Get frame metadata and create annotations
                    meta = client.get_task_data_meta(new_task_id)
                    frames_info = meta.get("frames", [])

                    filename_to_frame = {}
                    for frame_num, frame_info in enumerate(frames_info):
                        fname = frame_info.get("name", "")
                        filename_to_frame[fname] = frame_num
                        basename = os.path.basename(fname)
                        if basename != fname:
                            filename_to_frame[basename] = frame_num

                    # Get label_id
                    labels = client.get_labels(project_id=job.target_project_id)
                    label_id = None
                    for label in labels:
                        if label.get("name") == job.label_name:
                            label_id = label.get("id")
                            break

                    if label_id is None:
                        raise ValueError(f"Label '{job.label_name}' not found")

                    # Create annotations
                    shapes = []
                    group_id_counter = 1

                    for meta_item in file_metadata:
                        filename = meta_item["filename"]
                        item = meta_item["item"]
                        expected_frame = meta_item["expected_frame"]

                        frame_num = filename_to_frame.get(filename, expected_frame)

                        masks = item["masks"]
                        if hasattr(masks, 'cpu'):
                            masks = masks.cpu().numpy()
                        if masks.ndim == 4:
                            masks = masks.squeeze(1)

                        for mask in masks:
                            polygons = client.mask_to_polygon(
                                mask,
                                merge_regions=job.merge_regions,
                                merge_kernel_size=job.merge_kernel_size,
                                merge_method=job.merge_method
                            )
                            current_group_id = group_id_counter if len(polygons) > 1 else 0
                            if len(polygons) > 1:
                                group_id_counter += 1

                            for polygon_points in polygons:
                                if len(polygon_points) >= 6:
                                    shapes.append({
                                        "type": "polygon",
                                        "occluded": False,
                                        "z_order": 0,
                                        "points": polygon_points,
                                        "frame": frame_num,
                                        "label_id": label_id,
                                        "group": current_group_id,
                                        "attributes": []
                                    })

                    # Upload annotations
                    if shapes:
                        annotations = {
                            "version": 0,
                            "tags": [],
                            "shapes": shapes,
                            "tracks": []
                        }
                        url = f"{client.base_url}/api/tasks/{new_task_id}/annotations"
                        response = client.session.put(url, headers=client._get_headers(), json=annotations, timeout=60)
                        response.raise_for_status()

                    total_annotations += len(shapes)
                    task_info = {
                        "task_id": new_task_id,
                        "task_name": task_name,
                        "split": split_name,
                        "images": len(chunk),
                        "annotations": len(shapes)
                    }
                    created_tasks.append(task_info)
                    job.progress.created_tasks.append(task_info)
                    job.add_log(f"✓ 建立任務 {task_name}: {len(chunk)} 張圖片, {len(shapes)} 個標註")

                except Exception as e:
                    job.add_log(f"✗ 建立任務失敗 {task_name}: {e}")
                    logger.error(f"Error creating task {task_name}: {e}")
                    continue

        # Generate summary
        train_tasks = [t for t in created_tasks if t.get('split') == 'train']
        test_tasks = [t for t in created_tasks if t.get('split') == 'test']
        val_tasks = [t for t in created_tasks if t.get('split') == 'val']

        job.result_summary = f"""
## ✅ 批次處理完成!

### 掃描結果
- **來源專案**: {source_project['name']}
- **掃描任務數**: {total_tasks}
- **掃描總幀數**: {total_frames}
- **檢測到物件**: {len(detected_images)} 張

### 資料集分割
- **Train**: {sum(t['images'] for t in train_tasks)} 張 ({len(train_tasks)} 個任務)
- **Test**: {sum(t['images'] for t in test_tasks)} 張 ({len(test_tasks)} 個任務)
- **Validation**: {sum(t['images'] for t in val_tasks)} 張 ({len(val_tasks)} 個任務)

### 輸出結果
- **建立任務數**: {len(created_tasks)}
- **總標註數**: {total_annotations}
"""

    except Exception as e:
        job.error_message = str(e)
        job.add_log(f"任務失敗: {e}")
        raise


def start_background_batch_job(
    source_project_id: int,
    target_project_id: int,
    prompt: str,
    label_name: str,
    confidence_threshold: float,
    images_per_task: int,
    train_ratio: float,
    test_ratio: float,
    val_ratio: float,
    merge_regions: bool,
    merge_kernel_size: int,
    merge_method: str,
):
    """Start a batch job in the background."""
    if not source_project_id:
        return "❌ 請選擇來源專案", None
    if not target_project_id:
        return "❌ 請選擇目標專案", None
    if not prompt:
        return "❌ 請輸入搜尋物件", None
    if not label_name:
        return "❌ 請選擇標籤", None

    try:
        client = get_cvat_client()
        source_project = client.get_project(source_project_id)
        target_project = client.get_project(target_project_id)

        manager = get_job_manager()
        job = manager.create_job(
            source_project_id=source_project_id,
            source_project_name=source_project['name'],
            target_project_id=target_project_id,
            target_project_name=target_project['name'],
            prompt=prompt,
            label_name=label_name,
            confidence_threshold=confidence_threshold,
            images_per_task=int(images_per_task),
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            merge_regions=merge_regions,
            merge_kernel_size=int(merge_kernel_size),
            merge_method=merge_method,
        )

        manager.start_job(job.job_id, run_batch_job_background)

        return f"""
## ✅ 後台任務已啟動!

- **任務 ID**: `{job.job_id}`
- **來源專案**: {source_project['name']}
- **目標專案**: {target_project['name']}
- **搜尋物件**: {prompt}

**可以關閉此頁面，任務會在後台繼續執行。**
**到「📋 任務監控」頁籤查看進度。**
""", job.job_id

    except Exception as e:
        logger.error(f"Failed to start background job: {e}")
        return f"❌ 啟動失敗: {str(e)}", None


def get_job_status_display():
    """Get formatted job status for display."""
    manager = get_job_manager()
    jobs = manager.get_all_jobs()

    if not jobs:
        return "目前沒有任何任務", [], ""

    status_lines = []
    job_choices = []

    for job in jobs[:20]:  # Show last 20 jobs
        status_emoji = {
            JobStatus.PENDING: "⏳",
            JobStatus.RUNNING: "🔄",
            JobStatus.COMPLETED: "✅",
            JobStatus.FAILED: "❌",
            JobStatus.CANCELLED: "⏹️",
        }.get(job.status, "❓")

        progress_text = ""
        if job.status == JobStatus.RUNNING:
            progress_text = f" ({job.progress.percentage:.1f}%)"

        status_lines.append(
            f"{status_emoji} `{job.job_id}` - {job.prompt} → {job.target_project_name}{progress_text}"
        )
        job_choices.append((f"{job.job_id} - {job.prompt}", job.job_id))

    return "\n".join(status_lines), job_choices, ""


def get_job_detail(job_id: str):
    """Get detailed information for a specific job."""
    if not job_id:
        return "請選擇一個任務", ""

    manager = get_job_manager()
    job = manager.get_job(job_id)

    if not job:
        return "找不到此任務", ""

    status_emoji = {
        JobStatus.PENDING: "⏳ 等待中",
        JobStatus.RUNNING: "🔄 執行中",
        JobStatus.COMPLETED: "✅ 已完成",
        JobStatus.FAILED: "❌ 失敗",
        JobStatus.CANCELLED: "⏹️ 已取消",
    }.get(job.status, "❓ 未知")

    detail = f"""
## 任務詳情: `{job.job_id}`

### 狀態: {status_emoji}

### 設定
- **來源專案**: {job.source_project_name} (ID: {job.source_project_id})
- **目標專案**: {job.target_project_name} (ID: {job.target_project_id})
- **搜尋物件**: {job.prompt}
- **標籤**: {job.label_name}
- **信心度**: {job.confidence_threshold}
- **每任務圖片數**: {job.images_per_task}
- **Train/Test/Val 比例**: {job.train_ratio}/{job.test_ratio}/{job.val_ratio}

### 進度
- **已處理幀數**: {job.progress.processed_frames} / {job.progress.total_frames}
- **進度百分比**: {job.progress.percentage:.1f}%
- **檢測到物件**: {job.progress.detected_count} 張
- **當前任務**: {job.progress.current_task}
- **已建立任務數**: {len(job.progress.created_tasks)}

### 時間
- **建立時間**: {job.created_at.strftime('%Y-%m-%d %H:%M:%S')}
- **開始時間**: {job.started_at.strftime('%Y-%m-%d %H:%M:%S') if job.started_at else 'N/A'}
- **完成時間**: {job.completed_at.strftime('%Y-%m-%d %H:%M:%S') if job.completed_at else 'N/A'}
"""

    if job.error_message:
        detail += f"\n### ❌ 錯誤訊息\n```\n{job.error_message}\n```\n"

    if job.result_summary:
        detail += f"\n{job.result_summary}\n"

    # Created tasks
    if job.progress.created_tasks:
        detail += "\n### 已建立的任務\n"
        for t in job.progress.created_tasks:
            detail += f"- [{t['task_name']}](http://192.168.50.15:8080/tasks/{t['task_id']}) - {t['images']} 張圖片, {t['annotations']} 個標註\n"

    log_text = "\n".join(job.log_messages[-30:])

    return detail, log_text


def cancel_background_job(job_id: str):
    """Cancel a running background job."""
    if not job_id:
        return "請選擇一個任務"

    manager = get_job_manager()
    if manager.cancel_job(job_id):
        return f"✅ 已發送取消請求給任務 `{job_id}`"
    else:
        return f"❌ 無法取消任務 `{job_id}` (可能已經完成或不存在)"


def delete_background_job(job_id: str):
    """Delete a completed/failed job."""
    if not job_id:
        return "請選擇一個任務", [], ""

    manager = get_job_manager()
    if manager.delete_job(job_id):
        return f"✅ 已刪除任務 `{job_id}`", *get_job_status_display()[:2]
    else:
        return f"❌ 無法刪除任務 `{job_id}` (可能正在執行中)", *get_job_status_display()[:2]


# Gradio UI handlers
def on_project_select(project_id):
    """Handle project selection."""
    if not project_id:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None), None, ""

    tasks = load_tasks(project_id)
    return gr.update(choices=tasks, value=None), gr.update(choices=[], value=None), None, ""


def on_task_select(task_id):
    """Handle task selection."""
    if not task_id:
        return gr.update(choices=[], value=None), None, ""

    frames, info = load_frames(task_id)
    info_text = f"總幀數: {info.get('total', 0)}"
    return gr.update(choices=frames, value=None), None, info_text


def on_frame_select(task_id, frame_number):
    """Handle frame selection."""
    if task_id is None or frame_number is None:
        return None

    image = get_frame_image(task_id, frame_number)
    if image:
        return np.array(image)
    return None


def on_detect(task_id, frame_number, prompt, confidence):
    """Handle detection button click."""
    if task_id is None or frame_number is None or not prompt:
        return None, "請選擇任務、幀並輸入搜尋物件"

    image = get_frame_image(task_id, frame_number)
    if image is None:
        return None, "無法載入圖片"

    result_image, info = detect_objects(image, prompt, confidence)

    if "error" in info:
        return result_image, f"錯誤: {info['error']}"

    result_text = f"""
**搜尋物件**: {prompt}
**結果**: {info['message']}
"""
    if info.get("detected") and info.get("scores"):
        result_text += "\n**信心度**:\n"
        for i, score in enumerate(info["scores"]):
            result_text += f"  - 物件 {i+1}: {score*100:.1f}%\n"

    return result_image, result_text


def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(
        title="CVAT-SAM3 物件檢測"
    ) as demo:
        gr.Markdown("""
# CVAT-SAM3 物件檢測系統

從 CVAT 專案中選擇圖片，使用 SAM3 模型檢測指定物件。
        """)

        with gr.Tabs():
            # Tab 1: Single Frame Detection
            with gr.TabItem("📷 單張檢測"):
                with gr.Row():
                    # Left panel - Selection
                    with gr.Column(scale=1):
                        gr.Markdown("### 選擇資料")

                        # Refresh button
                        refresh_btn = gr.Button("🔄 重新載入專案", size="sm")

                        # Project dropdown
                        project_dropdown = gr.Dropdown(
                            label="選擇專案",
                            choices=[],
                            interactive=True,
                            allow_custom_value=False
                        )

                        # Task dropdown
                        task_dropdown = gr.Dropdown(
                            label="選擇任務",
                            choices=[],
                            interactive=True,
                            allow_custom_value=False
                        )

                        # Task info
                        task_info = gr.Textbox(label="任務資訊", interactive=False)

                        # Frame dropdown
                        frame_dropdown = gr.Dropdown(
                            label="選擇幀",
                            choices=[],
                            interactive=True,
                            allow_custom_value=False
                        )

                        gr.Markdown("### 檢測設定")

                        # Prompt input
                        prompt_input = gr.Textbox(
                            label="搜尋物件",
                            placeholder="輸入要搜尋的物件 (例如: person, car, crane)",
                            value=""
                        )

                        # Confidence slider
                        confidence_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.95,
                            value=0.9,
                            step=0.05,
                            label="信心度閾值"
                        )

                        # Detect button
                        detect_btn = gr.Button("🔍 檢測物件", variant="primary")

                    # Right panel - Results
                    with gr.Column(scale=2):
                        gr.Markdown("### 檢測結果")

                        # Result image
                        result_image = gr.Image(
                            label="結果圖片",
                            type="numpy",
                            height=500
                        )

                        # Result text
                        result_text = gr.Markdown(label="檢測資訊")

                # Upload to CVAT section
                gr.Markdown("---")
                gr.Markdown("### 📤 上傳檢測結果到 CVAT")

                with gr.Row():
                    with gr.Column(scale=1):
                        # Target project for upload
                        upload_project_dropdown = gr.Dropdown(
                            label="目標專案",
                            choices=[],
                            interactive=True,
                            allow_custom_value=False
                        )

                        # Label selection
                        label_dropdown = gr.Dropdown(
                            label="標籤名稱",
                            choices=[],
                            interactive=True,
                            allow_custom_value=False
                        )

                        # Task name input
                        new_task_name = gr.Textbox(
                            label="新任務名稱",
                            placeholder="輸入新任務的名稱",
                            value=""
                        )

                        # Upload button
                        upload_btn = gr.Button("📤 上傳到 CVAT", variant="primary", interactive=False)

                    with gr.Column(scale=2):
                        upload_result = gr.Markdown(label="上傳結果")

                # Batch analysis section
                gr.Markdown("---")
                gr.Markdown("### 批次分析")

                with gr.Row():
                    with gr.Column(scale=1):
                        max_frames_slider = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=10,
                            step=1,
                            label="最大分析幀數"
                        )
                        batch_analyze_btn = gr.Button("📊 批次分析任務", variant="secondary")

                    with gr.Column(scale=2):
                        batch_summary = gr.Markdown(label="分析摘要")

                # Gallery for batch results
                batch_gallery = gr.Gallery(
                    label="包含物件的幀",
                    columns=4,
                    height=300,
                    object_fit="contain"
                )

            # Tab 2: Batch Project Scan
            with gr.TabItem("🔄 批次專案掃描"):
                gr.Markdown("""
### 批次專案掃描與轉換

掃描來源專案中的所有任務和圖片，自動檢測指定物件，並將包含該物件的圖片轉移到目標專案。
每個新任務最多包含指定數量的圖片。
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 來源設定")

                        # Refresh for batch tab
                        batch_refresh_btn = gr.Button("🔄 重新載入專案", size="sm")

                        # Source project
                        batch_source_project = gr.Dropdown(
                            label="來源專案",
                            choices=[],
                            interactive=True,
                            allow_custom_value=False
                        )

                        # Source project info
                        batch_source_info = gr.Markdown("選擇專案後顯示資訊")

                        gr.Markdown("#### 目標設定")

                        # Target project
                        batch_target_project = gr.Dropdown(
                            label="目標專案",
                            choices=[],
                            interactive=True,
                            allow_custom_value=False
                        )

                        # Target label
                        batch_label_dropdown = gr.Dropdown(
                            label="標籤名稱",
                            choices=[],
                            interactive=True,
                            allow_custom_value=False
                        )

                        gr.Markdown("#### 檢測設定")

                        # Prompt
                        batch_prompt = gr.Textbox(
                            label="搜尋物件",
                            placeholder="輸入要搜尋的物件 (例如: person, car, crane)",
                            value=""
                        )

                        # Confidence
                        batch_confidence = gr.Slider(
                            minimum=0.1,
                            maximum=0.95,
                            value=0.9,
                            step=0.05,
                            label="信心度閾值"
                        )

                        # Images per task (ZIP compression allows larger batches)
                        images_per_task = gr.Slider(
                            minimum=50,
                            maximum=1000,
                            value=500,
                            step=50,
                            label="每個任務最大圖片數 (ZIP壓縮上傳)"
                        )

                        gr.Markdown("#### 資料集分割設定")
                        gr.Markdown("*將檢測結果分成 Train/Test/Validation 三組*")

                        with gr.Row():
                            train_ratio = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=70,
                                step=5,
                                label="Train %"
                            )
                            test_ratio = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=20,
                                step=5,
                                label="Test %"
                            )
                            val_ratio = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=10,
                                step=5,
                                label="Validation %"
                            )

                        ratio_warning = gr.Markdown("*總和: 100% ✓*")

                        gr.Markdown("#### 區域融合設定")
                        gr.Markdown("*當物件被遮擋分成多個區域時，可以選擇融合它們*")

                        merge_checkbox = gr.Checkbox(
                            label="啟用區域融合",
                            value=False,
                            info="將同一物件的分離區域融合成一個 polygon"
                        )

                        merge_method_dropdown = gr.Dropdown(
                            label="融合方法",
                            choices=[
                                ("閉運算 (推薦)", "closing"),
                                ("膨脹", "dilate"),
                                ("凸包", "convex")
                            ],
                            value="closing",
                            info="closing: 保持原始大小; dilate: 區域會變大; convex: 包圍所有區域"
                        )

                        merge_kernel_slider = gr.Slider(
                            minimum=5,
                            maximum=300,
                            value=15,
                            step=5,
                            label="融合強度 (kernel size)",
                            info="越大可以連接越遠的分離區域"
                        )

                        with gr.Row():
                            batch_start_btn = gr.Button("🚀 開始批次掃描", variant="primary")
                            batch_stop_btn = gr.Button("⏹️ 停止", variant="stop")

                    with gr.Column(scale=2):
                        gr.Markdown("#### 執行狀態")

                        # Status
                        batch_status = gr.Markdown("等待開始...")

                        # Log
                        batch_log = gr.Textbox(
                            label="執行日誌",
                            lines=10,
                            interactive=False
                        )

                        # Results gallery (optional preview)
                        batch_preview = gr.Gallery(
                            label="檢測預覽",
                            columns=4,
                            height=200,
                            object_fit="contain"
                        )

            # Tab 3: Job Monitor
            with gr.TabItem("📋 任務監控"):
                gr.Markdown("""
### 後台任務監控

查看和管理後台執行的批次任務。**關閉網頁後任務仍會繼續執行**，重新開啟此頁面即可查看進度。
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 任務列表")

                        monitor_refresh_btn = gr.Button("🔄 重新整理", size="sm")

                        job_list_display = gr.Markdown("載入中...")

                        job_selector = gr.Dropdown(
                            label="選擇任務查看詳情",
                            choices=[],
                            interactive=True,
                            allow_custom_value=False
                        )

                        with gr.Row():
                            cancel_job_btn = gr.Button("⏹️ 取消任務", variant="stop", size="sm")
                            delete_job_btn = gr.Button("🗑️ 刪除記錄", size="sm")

                        cancel_result = gr.Markdown("")

                    with gr.Column(scale=2):
                        gr.Markdown("#### 任務詳情")

                        job_detail_display = gr.Markdown("選擇任務查看詳情")

                        job_log_display = gr.Textbox(
                            label="執行日誌",
                            lines=15,
                            interactive=False
                        )

                        # Auto-refresh timer
                        auto_refresh = gr.Checkbox(
                            label="自動重新整理 (每 5 秒)",
                            value=False
                        )

        # Event handlers
        def refresh_projects():
            projects = load_projects()
            return gr.update(choices=projects, value=None), gr.update(choices=projects, value=None)

        def refresh_all_projects():
            """Refresh projects for all dropdowns."""
            projects = load_projects()
            return (
                gr.update(choices=projects, value=None),
                gr.update(choices=projects, value=None),
                gr.update(choices=projects, value=None),
                gr.update(choices=projects, value=None)
            )

        def on_batch_source_select(project_id):
            """Handle batch source project selection."""
            if not project_id:
                return "選擇專案後顯示資訊"

            try:
                client = get_cvat_client()
                project = client.get_project(project_id)
                tasks = client.get_project_tasks(project_id)
                total_frames = sum(t.get("size", 0) for t in tasks)

                return f"""
**專案名稱**: {project['name']}
**任務數量**: {len(tasks)}
**總幀數**: {total_frames}
"""
            except Exception as e:
                return f"錯誤: {str(e)}"

        def update_ratio_warning(train, test, val):
            """Update the ratio warning message."""
            total = train + test + val
            if total == 100:
                return "*總和: 100% ✓*"
            elif total == 0:
                return "*⚠️ 總和: 0% (將使用預設 70/20/10)*"
            else:
                return f"*⚠️ 總和: {total}% (將自動正規化)*"

        # Single frame tab events
        refresh_btn.click(
            fn=refresh_projects,
            outputs=[project_dropdown, upload_project_dropdown]
        )

        project_dropdown.change(
            fn=on_project_select,
            inputs=[project_dropdown],
            outputs=[task_dropdown, frame_dropdown, result_image, result_text]
        )

        task_dropdown.change(
            fn=on_task_select,
            inputs=[task_dropdown],
            outputs=[frame_dropdown, result_image, task_info]
        )

        frame_dropdown.change(
            fn=on_frame_select,
            inputs=[task_dropdown, frame_dropdown],
            outputs=[result_image]
        )

        detect_btn.click(
            fn=detect_and_store,
            inputs=[task_dropdown, frame_dropdown, prompt_input, confidence_slider],
            outputs=[result_image, result_text, upload_btn]
        )

        upload_project_dropdown.change(
            fn=load_project_labels,
            inputs=[upload_project_dropdown],
            outputs=[label_dropdown]
        )

        upload_btn.click(
            fn=upload_to_cvat,
            inputs=[upload_project_dropdown, new_task_name, label_dropdown],
            outputs=[upload_result]
        )

        batch_analyze_btn.click(
            fn=analyze_task_frames,
            inputs=[task_dropdown, prompt_input, confidence_slider, max_frames_slider],
            outputs=[batch_summary, batch_gallery]
        )

        # Batch tab events
        batch_refresh_btn.click(
            fn=refresh_all_projects,
            outputs=[project_dropdown, upload_project_dropdown, batch_source_project, batch_target_project]
        )

        batch_source_project.change(
            fn=on_batch_source_select,
            inputs=[batch_source_project],
            outputs=[batch_source_info]
        )

        batch_target_project.change(
            fn=load_project_labels,
            inputs=[batch_target_project],
            outputs=[batch_label_dropdown]
        )

        # Ratio validation events
        train_ratio.change(
            fn=update_ratio_warning,
            inputs=[train_ratio, test_ratio, val_ratio],
            outputs=[ratio_warning]
        )
        test_ratio.change(
            fn=update_ratio_warning,
            inputs=[train_ratio, test_ratio, val_ratio],
            outputs=[ratio_warning]
        )
        val_ratio.change(
            fn=update_ratio_warning,
            inputs=[train_ratio, test_ratio, val_ratio],
            outputs=[ratio_warning]
        )

        # Background job mode for batch scan
        batch_start_btn.click(
            fn=start_background_batch_job,
            inputs=[
                batch_source_project,
                batch_target_project,
                batch_prompt,
                batch_label_dropdown,
                batch_confidence,
                images_per_task,
                train_ratio,
                test_ratio,
                val_ratio,
                merge_checkbox,
                merge_kernel_slider,
                merge_method_dropdown
            ],
            outputs=[batch_status, batch_log]
        )

        batch_stop_btn.click(
            fn=stop_batch_scan,
            outputs=[batch_status]
        )

        # Job monitor tab events
        def refresh_job_list():
            status, choices, _ = get_job_status_display()
            return status, gr.update(choices=choices)

        def on_job_select(job_id):
            if not job_id:
                return "選擇任務查看詳情", ""
            return get_job_detail(job_id)

        def refresh_selected_job(job_id):
            """Refresh details for the currently selected job."""
            if not job_id:
                status, choices, _ = get_job_status_display()
                return status, gr.update(choices=choices), "選擇任務查看詳情", ""
            status, choices, _ = get_job_status_display()
            detail, log = get_job_detail(job_id)
            return status, gr.update(choices=choices), detail, log

        monitor_refresh_btn.click(
            fn=refresh_selected_job,
            inputs=[job_selector],
            outputs=[job_list_display, job_selector, job_detail_display, job_log_display]
        )

        job_selector.change(
            fn=on_job_select,
            inputs=[job_selector],
            outputs=[job_detail_display, job_log_display]
        )

        cancel_job_btn.click(
            fn=cancel_background_job,
            inputs=[job_selector],
            outputs=[cancel_result]
        )

        delete_job_btn.click(
            fn=delete_background_job,
            inputs=[job_selector],
            outputs=[cancel_result, job_list_display, job_selector]
        )

        # Auto-refresh functionality
        def auto_refresh_handler(is_enabled, job_id):
            if is_enabled and job_id:
                return get_job_detail(job_id)
            return gr.update(), gr.update()

        # Use timer for auto-refresh (every 5 seconds when enabled)
        auto_refresh_timer = gr.Timer(5, active=False)

        auto_refresh.change(
            fn=lambda x: gr.Timer(active=x),
            inputs=[auto_refresh],
            outputs=[auto_refresh_timer]
        )

        auto_refresh_timer.tick(
            fn=refresh_selected_job,
            inputs=[job_selector],
            outputs=[job_list_display, job_selector, job_detail_display, job_log_display]
        )

        # Load projects and job list on startup
        def load_all_on_startup():
            projects = load_projects()
            status, choices, _ = get_job_status_display()
            return (
                gr.update(choices=projects, value=None),
                gr.update(choices=projects, value=None),
                gr.update(choices=projects, value=None),
                gr.update(choices=projects, value=None),
                status,
                gr.update(choices=choices)
            )

        demo.load(
            fn=load_all_on_startup,
            outputs=[
                project_dropdown,
                upload_project_dropdown,
                batch_source_project,
                batch_target_project,
                job_list_display,
                job_selector
            ]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7861")),
        share=False,
    )
