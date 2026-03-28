import time
import base64
import cv2
from PIL import Image
import tempfile
import requests
from typing import Literal
from runwayml import RunwayML

def extract_frames_from_url(video_url: str):
    frames = []
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as tmp_file:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tmp_file.write(chunk)
        tmp_file.flush()
        
        cap = cv2.VideoCapture(tmp_file.name)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)
                
        finally:
            cap.release()
    
    return frames

class Gen3:
    def __init__(
        self,
        model_name: str = "gen_3_i2v",
        generation_type: Literal["t2v", "i2v", "v2v"] = "i2v",
        model_id: str = "gen3a_turbo",
        inference: dict = None,
        helper_config: dict = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.generation_type = generation_type
        self.model_id = model_id
        
        # Extract configuration from inference section
        inference = inference or {}
        helper_config = helper_config or {}
        
        # Get configuration values with defaults
        self.resolution = inference.get("resolution", [1280, 768])
        self.ratio = inference.get("ratio", "1280:768")
        self.duration = inference.get("duration", 5)
        
        # Store configs for potential future use
        self.inference_config = inference
        self.helper_config = helper_config
        
        self.client = RunwayML()
        
    def generate_video(
        self,
        prompt: str,
        image_path: str,
    ):    

        # encode image to base64
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        # Create a new image-to-video task using the configured model
        task = self.client.image_to_video.create(
            model=self.model_id,
            # Point this at your own image file
            prompt_image=f"data:image/png;base64,{base64_image}",
            prompt_text=prompt,
            ratio=self.ratio,
            duration=self.duration,
        )
        task_id = task.id

        # Poll the task until it's complete
        time.sleep(10)  # Wait for a second before polling
        task = self.client.tasks.retrieve(task_id)
        while task.status not in ['SUCCEEDED', 'FAILED']:
            time.sleep(10)  # Wait for ten seconds before polling
            task = self.client.tasks.retrieve(task_id)

        print('Task complete:', task)
        video_url = task.output[0]
        frames = extract_frames_from_url(video_url)

        return frames
