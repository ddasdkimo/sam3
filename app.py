"""
SAM3 Gradio Web Interface
Segment Anything with Concepts - Interactive Demo
"""

import gc
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

# SAM3 imports
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor


# Global model instances (lazy loading)
_image_model = None
_image_processor = None
_video_predictor = None


def get_image_model():
    """Lazy load image model."""
    global _image_model, _image_processor
    if _image_model is None:
        print("Loading SAM3 image model...")
        _image_model = build_sam3_image_model(
            device="cuda" if torch.cuda.is_available() else "cpu",
            eval_mode=True,
            load_from_HF=True,
        )
        _image_processor = Sam3Processor(
            _image_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("SAM3 image model loaded successfully!")
    return _image_model, _image_processor


def get_video_predictor():
    """Lazy load video predictor."""
    global _video_predictor
    if _video_predictor is None:
        print("Loading SAM3 video predictor...")
        _video_predictor = build_sam3_video_predictor()
        print("SAM3 video predictor loaded successfully!")
    return _video_predictor


def create_color_mask(mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.5) -> np.ndarray:
    """Create a colored mask overlay."""
    colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
    colored_mask[mask > 0] = [*color, int(alpha * 255)]
    return colored_mask


def overlay_masks_on_image(image: np.ndarray, masks: np.ndarray, scores: Optional[np.ndarray] = None) -> np.ndarray:
    """Overlay segmentation masks on the image."""
    if len(masks) == 0:
        return image

    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    result = image.copy().astype(np.float32)

    # Color palette for different instances
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 255, 128),  # Spring Green
        (255, 0, 128),  # Rose
    ]

    alpha = 0.5
    for i, mask in enumerate(masks):
        if mask.ndim > 2:
            mask = mask.squeeze()

        color = colors[i % len(colors)]
        mask_bool = mask > 0.5

        # Apply colored overlay
        for c in range(3):
            result[:, :, c] = np.where(
                mask_bool,
                result[:, :, c] * (1 - alpha) + color[c] * alpha,
                result[:, :, c]
            )

        # Draw contour
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color, 2)

    return result.astype(np.uint8)


def process_image(image: np.ndarray, prompt: str, confidence_threshold: float) -> Tuple[np.ndarray, str]:
    """Process a single image with SAM3."""
    if image is None:
        return None, "Please upload an image."

    if not prompt or prompt.strip() == "":
        return image, "Please enter a text prompt."

    try:
        model, processor = get_image_model()

        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Set confidence threshold
        processor.confidence_threshold = confidence_threshold

        # Process image
        state = processor.set_image(pil_image)
        output = processor.set_text_prompt(prompt=prompt, state=state)

        # Get results
        masks = output.get("masks", None)
        scores = output.get("scores", None)
        boxes = output.get("boxes", None)

        if masks is None or len(masks) == 0:
            return image, f"No objects found matching '{prompt}'"

        # Convert tensors to numpy
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()

        # Squeeze extra dimensions
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Create result image
        result_image = overlay_masks_on_image(np.array(pil_image), masks, scores)

        # Create info text
        info_text = f"Found {len(masks)} object(s) matching '{prompt}'\n"
        if scores is not None:
            for i, score in enumerate(scores):
                info_text += f"  - Object {i+1}: confidence {score:.2%}\n"

        return result_image, info_text

    except Exception as e:
        import traceback
        error_msg = f"Error processing image: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return image, error_msg


def process_video(video_path: str, prompt: str, progress=gr.Progress()) -> Tuple[Optional[str], str]:
    """Process a video with SAM3."""
    if video_path is None:
        return None, "Please upload a video."

    if not prompt or prompt.strip() == "":
        return None, "Please enter a text prompt."

    try:
        predictor = get_video_predictor()

        # Generate unique session ID
        session_id = str(uuid.uuid4())

        progress(0.1, desc="Starting video session...")

        # Start session
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
                session_id=session_id,
            )
        )

        progress(0.2, desc="Adding text prompt...")

        # Add prompt on first frame
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=prompt,
            )
        )

        progress(0.3, desc="Propagating through video...")

        # Collect all frame outputs
        frame_outputs = {}
        for frame_result in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="forward",
            )
        ):
            frame_idx = frame_result["frame_index"]
            frame_outputs[frame_idx] = frame_result["outputs"]

        progress(0.7, desc="Rendering output video...")

        # Read original video and create output
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output video
        output_path = os.path.join(tempfile.gettempdir(), f"sam3_output_{session_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get masks for this frame
            if frame_idx in frame_outputs:
                outputs = frame_outputs[frame_idx]
                if outputs and len(outputs) > 0:
                    # Collect all masks for this frame
                    all_masks = []
                    for obj_id, obj_output in outputs.items():
                        mask = obj_output.get("mask", None)
                        if mask is not None:
                            if isinstance(mask, torch.Tensor):
                                mask = mask.cpu().numpy()
                            # Resize mask to frame size if needed
                            if mask.shape[:2] != (height, width):
                                mask = cv2.resize(mask.astype(np.float32), (width, height))
                            all_masks.append(mask)

                    if all_masks:
                        masks_array = np.array(all_masks)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = overlay_masks_on_image(frame_rgb, masks_array)
                        frame = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            out.write(frame)
            frame_idx += 1

            if frame_idx % 10 == 0:
                progress(0.7 + 0.25 * (frame_idx / total_frames), desc=f"Processing frame {frame_idx}/{total_frames}")

        cap.release()
        out.release()

        # Close session
        predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )

        progress(1.0, desc="Done!")

        # Clean up GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        info_text = f"Video processed successfully!\n"
        info_text += f"Total frames: {total_frames}\n"
        info_text += f"Objects tracked: {len(frame_outputs.get(0, {}))}\n"

        return output_path, info_text

    except Exception as e:
        import traceback
        error_msg = f"Error processing video: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


# Create Gradio interface
def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(
        title="SAM3 - Segment Anything with Concepts",
    ) as demo:
        gr.Markdown(
            """
            # SAM3: Segment Anything with Concepts

            Upload an image or video and enter a text prompt to segment objects.
            SAM3 can detect, segment, and track objects using natural language descriptions.

            **Examples of prompts:**
            - "a person wearing red"
            - "the dog"
            - "cars on the road"
            - "a cup on the table"
            """
        )

        with gr.Tabs():
            # Image Tab
            with gr.TabItem("Image Segmentation"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Upload Image",
                            type="numpy",
                        )
                        image_prompt = gr.Textbox(
                            label="Text Prompt",
                            placeholder="Enter what you want to segment (e.g., 'a person', 'the cat', 'red car')",
                        )
                        image_confidence = gr.Slider(
                            minimum=0.1,
                            maximum=0.95,
                            value=0.5,
                            step=0.05,
                            label="Confidence Threshold",
                        )
                        image_btn = gr.Button("Segment Image", variant="primary")

                    with gr.Column():
                        image_output = gr.Image(label="Segmentation Result")
                        image_info = gr.Textbox(label="Information", lines=5)

                image_btn.click(
                    fn=process_image,
                    inputs=[image_input, image_prompt, image_confidence],
                    outputs=[image_output, image_info],
                )

                # Image examples
                gr.Examples(
                    examples=[
                        ["a person"],
                        ["the dog"],
                        ["cars"],
                        ["a red object"],
                    ],
                    inputs=[image_prompt],
                )

            # Video Tab
            with gr.TabItem("Video Segmentation"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(
                            label="Upload Video (MP4)",
                        )
                        video_prompt = gr.Textbox(
                            label="Text Prompt",
                            placeholder="Enter what you want to track (e.g., 'a person', 'the ball')",
                        )
                        video_btn = gr.Button("Process Video", variant="primary")

                    with gr.Column():
                        video_output = gr.Video(label="Segmentation Result")
                        video_info = gr.Textbox(label="Information", lines=5)

                video_btn.click(
                    fn=process_video,
                    inputs=[video_input, video_prompt],
                    outputs=[video_output, video_info],
                )

        gr.Markdown(
            """
            ---
            **Notes:**
            - First run will download model weights (~3GB) from Hugging Face
            - Requires GPU with CUDA support for optimal performance
            - Video processing may take several minutes depending on length

            **Model:** SAM3 (Segment Anything Model 3) by Meta AI Research
            """
        )

    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
        share=False,
    )
