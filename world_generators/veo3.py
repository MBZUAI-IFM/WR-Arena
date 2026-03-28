import os
import time
import tempfile
from typing import Literal, List, Tuple

import cv2
from PIL import Image

from google import genai
from google.genai import types

def extract_frames_from_mp4_path(mp4_path: str) -> List[Image.Image]:
    """Read an mp4 from disk and return frames as PIL Images."""
    frames: List[Image.Image] = []
    cap = cv2.VideoCapture(mp4_path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    finally:
        cap.release()
    return frames


class Veo3:
    # Veo accepts 16:9 or 9:16 aspect ratios; for our benchmark we keep 16:9.
    SUPPORTED_SIZES: List[Tuple[int, int]] = [
        (720, 1280),
        (1280, 720),
        (1024, 1792),
        (1792, 1024),
    ]

    SUPPORTED_VIDEO_DURATION: List[int] = [4, 8]  # seconds (per your requirement)

    def __init__(
        self,
        model_name: str = "veo3_i2v",
        model_version: str = "veo-3.1-fast-generate-preview",
        generation_type: Literal["t2v", "i2v", "v2v"] = "i2v",
        api_key_env: str = "VEO3_API_KEY",
        default_seconds: int = 4,
        poll_interval_s: int = 10,
        inference: dict = None,
        helper_config: dict = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.generation_type = generation_type
        self.default_seconds = default_seconds
        self.poll_interval_s = poll_interval_s
        
        # Extract configuration from inference section
        inference = inference or {}
        helper_config = helper_config or {}
        
        # Get resolution from inference config, fallback to default
        resolution = inference.get("resolution", [1280, 720])
        self.default_target_size = (resolution[0], resolution[1])
        
        # Store other configs for potential future use
        self.inference_config = inference
        self.helper_config = helper_config

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key. Set {api_key_env}.")

        self.client = genai.Client(api_key=api_key)

    # image prep helpers 
    def _choose_target_size(self, w: int, h: int):
        """Pick supported size closest in aspect ratio."""
        ar = w / h

        def ar_diff(size: Tuple[int, int]) -> float:
            return abs((size[0] / size[1]) - ar)

        return min(self.SUPPORTED_SIZES, key=ar_diff)

    def _center_crop_to_ar(self, img: Image.Image, target_ar: float):
        """Center-crop PIL image to target aspect ratio."""
        w, h = img.size
        cur_ar = w / h

        if cur_ar > target_ar:
            new_w = int(h * target_ar)
            left = (w - new_w) // 2
            return img.crop((left, 0, left + new_w, h))
        else:
            new_h = int(w / target_ar)
            top = (h - new_h) // 2
            return img.crop((0, top, w, top + new_h))

    def _prepare_reference_image(self, image_path: str):
        """
        Ensure image matches benchmark target size (default 1280x720).
        Returns: (prepared_path, was_temp)
        """
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        target_w, target_h = self.default_target_size
        if (w, h) == (target_w, target_h):
            return image_path, False

        target_ar = target_w / target_h
        cropped = self._center_crop_to_ar(img, target_ar)
        resized = cropped.resize((target_w, target_h), Image.LANCZOS)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        resized.save(tmp.name, format="PNG")
        tmp.close()

        return tmp.name, True


    def generate_video(
        self,
        prompt: str,
        image_path: str,
    ):

        duration = self.default_seconds
        if duration not in self.SUPPORTED_VIDEO_DURATION:
            raise ValueError(
                f"Veo3 duration must be one of {self.SUPPORTED_VIDEO_DURATION}, got {duration}"
            )

        prepared_path, is_temp_img = self._prepare_reference_image(image_path)

        try:
            genai_image = types.Image.from_file(location=prepared_path)

            operation = self.client.models.generate_videos(
                model=self.model_version,
                source=types.GenerateVideosSource(
                    prompt=prompt,
                    image=genai_image,
                ),
                config=types.GenerateVideosConfig(
                    duration_seconds=duration
                ),
            )

            # Poll until ready
            while not operation.done:
                time.sleep(self.poll_interval_s)
                operation = self.client.operations.get(operation)

            # Get first generated video + download it
            generated_video = operation.response.generated_videos[0]
            self.client.files.download(file=generated_video.video)

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_mp4:
                mp4_path = tmp_mp4.name
            generated_video.video.save(mp4_path)

            frames = extract_frames_from_mp4_path(mp4_path)
            os.remove(mp4_path)

            return frames

        finally:
            if is_temp_img:
                try:
                    os.remove(prepared_path)
                except OSError:
                    pass