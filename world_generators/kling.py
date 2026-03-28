import time, base64, tempfile, requests, jwt, cv2, os
from PIL import Image
from typing import Literal

def extract_frames_from_url(video_url):
    frames = []
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as tmp_file:
        r = requests.get(video_url, stream=True)
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                tmp_file.write(chunk)
        tmp_file.flush()
        cap = cv2.VideoCapture(tmp_file.name)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        cap.release()
    return frames

class KLING:
    def __init__(
        self,
        model_name: str = "kling",
        generation_type: Literal["i2v"] = "i2v",
        model_id: str = "kling-v1",
        duration: int = 5,
        inference: dict = None,
        helper_config: dict = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.generation_type = generation_type
        self.model_id = model_id
        self.duration = duration
        
        inference = inference or {}
        helper_config = helper_config or {}
        
        self.resolution = inference.get("resolution", [1280, 720])
        self.frame_num = inference.get("frame_num", 153)
        
        self.inference_config = inference
        self.helper_config = helper_config
        
        # Fixed API key environment variables
        self.ak = os.environ["KLING_API_KEY"]
        self.sk = os.environ["KLING_API_SECRET"]

    def generate_video(
        self,
        prompt: str,
        image_path: str,
    ):
        
        def generate_jwt():
            payload = {
                "iss": self.ak,
                "exp": int(time.time()) + 1800,
                "nbf": int(time.time()) - 5
            }
            return jwt.encode(payload, self.sk, algorithm="HS256", headers={"alg": "HS256", "typ": "JWT"})
        
        token = generate_jwt()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."

        payload = {
            "model_name": self.model_id,
            "duration": self.duration,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": base64_image
        }

        create_url = "https://api-singapore.klingai.com/v1/videos/image2video"
        response = requests.post(create_url, headers=headers, json=payload)
        response.raise_for_status()
        task_id = response.json()["data"]["task_id"]

        # poll
        query_url = f"https://api-singapore.klingai.com/v1/videos/image2video/{task_id}"
        while True:
            res = requests.get(query_url, headers=headers).json()
            status = res["data"]["task_status"]
            if status == "succeed":
                video_url = res["data"]["task_result"]["videos"][0]["url"]
                break
            elif status == "failed":
                raise RuntimeError("❌ Task failed: " + res["data"].get("task_status_msg", "Unknown error"))
            time.sleep(10)

        return extract_frames_from_url(video_url)
