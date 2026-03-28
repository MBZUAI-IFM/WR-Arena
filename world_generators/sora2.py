import os
import tempfile
from typing import Literal, List, Tuple

import cv2
from PIL import Image
from openai import OpenAI


def extract_frames_from_mp4_path(mp4_path: str):
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


class Sora2:
    # Commonly-available Sora2 sizes
    SUPPORTED_SIZES: List[Tuple[int, int]] = [
        (720, 1280),
        (1280, 720),
        (1024, 1792),
        (1792, 1024),
    ]
    
    SUPPORTED_VIDEO_DURATION: List[int] = [
        4,
        8, 
        12
    ] # In seconds

    def __init__(
        self,
        model_name: str = "sora-2",
        generation_type: Literal["t2v", "i2v", "v2v"] = "i2v",
        api_key_env: str = "SORA2_API_KEY",
        model_version: str = "sora-2",
        default_seconds: int = 4,
        inference: dict = None,
        **kwargs
    ):
        self.model_name = model_name or "sora-2"
        self.generation_type = generation_type
        self.model_version = model_version

        inf = inference or {}
        self.default_seconds = inf.get("default_seconds", default_seconds)
        res = inf.get("resolution", [1280, 720])
        self.resolution = (int(res[0]), int(res[1]))
        if self.resolution not in self.SUPPORTED_SIZES:
            raise ValueError(
                f"resolution {self.resolution} not in Sora2 SUPPORTED_SIZES: {self.SUPPORTED_SIZES}"
            )

        api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                f"Missing API key. Set {api_key_env} or OPENAI_API_KEY."
            )

        self.client = OpenAI(api_key=api_key)

    # ---------- image prep helpers ----------
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
            # too wide -> crop width
            new_w = int(h * target_ar)
            left = (w - new_w) // 2
            return img.crop((left, 0, left + new_w, h))
        else:
            # too tall -> crop height
            new_h = int(w / target_ar)
            top = (h - new_h) // 2
            return img.crop((0, top, w, top + new_h))

    def _prepare_reference_image(self, image_path: str):
        """
        Ensure image matches a supported size.
        Returns: (prepared_path, size_str, was_temp)
        """
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        if (w, h) in self.SUPPORTED_SIZES:
            return image_path, f"{w}x{h}", False

        target_w, target_h = self.resolution
        target_ar = target_w / target_h

        cropped = self._center_crop_to_ar(img, target_ar)
        resized = cropped.resize((target_w, target_h), Image.LANCZOS)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        resized.save(tmp.name, format="PNG")
        tmp.close()

        return tmp.name, f"{target_w}x{target_h}", True


    def generate_video(
        self,
        prompt: str,
        image_path: str,
    ):

        prepared_path, size_str, is_temp_img = self._prepare_reference_image(image_path)

        try:
            with open(prepared_path, "rb") as img_file:
                video = self.client.videos.create_and_poll(
                    model=self.model_version,       
                    prompt=prompt,
                    input_reference=img_file,      
                    seconds=self.default_seconds,   
                    size=size_str,                 
                    poll_interval_ms=1500,
                )

            if video.status != "completed":
                raise RuntimeError(f"Sora2 failed: {video.status} — {video.error}")

            # Download MP4 bytes
            binary = self.client.videos.download_content(video.id)

            # Write to temp mp4 for OpenCV to read
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_mp4:
                mp4_path = tmp_mp4.name
                # HttpxBinaryResponseContent supports .content and .read()
                tmp_mp4.write(binary.content)

            frames = extract_frames_from_mp4_path(mp4_path)
            os.remove(mp4_path)

            return frames

        finally:
            if is_temp_img:
                try:
                    os.remove(prepared_path)
                except OSError:
                    pass
