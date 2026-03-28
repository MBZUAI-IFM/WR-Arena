import os
import time
import requests
import json
import base64
from typing import Literal, List
import cv2
from PIL import Image
import tempfile
from io import BytesIO

def resize_if_small(image_path, min_edge_size=300):
    with Image.open(image_path) as img:
        width, height = img.size
        min_edge = min(width, height)

        if min_edge <= min_edge_size:
            scale = (min_edge_size + 20) / min_edge  
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return img_str


def extract_frames_from_url(video_url):
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

class Minimax:
    def __init__(
        self,
        model_name: str = "minimax",
        generation_type: Literal["t2v", "i2v", "v2v"] = "i2v",
        api_url: str = "https://api.minimax.io/v1/video_generation",
        model_id: str = "I2V-01-Director",
        inference: dict = None,
        helper_config: dict = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.generation_type = generation_type
        self.api_url = api_url
        self.model_id = model_id
        
        inference = inference or {}
        helper_config = helper_config or {}
        
        self.resolution = inference.get("resolution", [1072, 720])
        self.frame_num = inference.get("frame_num", 141)
        
        self.inference_config = inference
        self.helper_config = helper_config
        
        # Fixed API credentials from environment
        self.api_key = os.environ["MINIMAX_API_KEY"]
        self.group_id = os.environ["MINIMAX_GROUP_ID"]
        
    def invoke_video_generation(self, prompt: str, image_path: str):
        print("-----------------Submit video generation task-----------------")

        data = resize_if_small(image_path)
        
        payload = json.dumps({
            "prompt": prompt,
            "model": self.model_id,
            "first_frame_image":f"data:image/jpeg;base64,{data}"
        })
        headers = {
            'authorization': 'Bearer ' + self.api_key,
            'content-type': 'application/json',
        }

        response = requests.request("POST", self.api_url, headers=headers, data=payload)
        print(response.text)
        task_id = response.json()['task_id']
        print("Video generation task submitted successfully, task ID:"+task_id)
        return task_id

    def query_video_generation(self, task_id: str):
        url = "https://api.minimax.io/v1/query/video_generation?task_id="+task_id
        headers = {
            'authorization': 'Bearer ' + self.api_key
        }
        response = requests.request("GET", url, headers=headers)
        status = response.json()['status']
        print(response.json())
        if status == 'Preparing':
            print("...Preparing...")
            return "", 'Preparing'   
        elif status == 'Queueing':
            print("...In the queue...")
            return "", 'Queueing'
        elif status == 'Processing':
            print("...Generating...")
            return "", 'Processing'
        elif status == 'Success':
            return response.json()['file_id'], "Finished"
        elif status == 'Fail':
            return "", "Fail"
        else:
            return "", "Unknown"
    
    def fetch_video_result(self, file_id: str):
        print("---------------Video generated successfully, downloading now---------------")
        url = "https://api.minimax.io/v1/files/retrieve?GroupId={}&file_id={}".format(self.group_id, file_id)

        headers = {
            'content-type': 'application/json',
            'Authorization': 'Bearer '+ self.api_key,
        }

        response = requests.request("GET", url, headers=headers)
        print(response.text)

        download_url = response.json()['file']['download_url']
        print("Video download link:" + download_url)
        frames = extract_frames_from_url(download_url)
        return frames
    
    def generate_video(
        self, 
        prompt: str, 
        image_path: str,
        max_retries: int = 1
    ):
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            print(f"=== Attempt {attempt} to generate video ===")
            task_id = self.invoke_video_generation(prompt, image_path)
            print("-----------------Video generation task submitted -----------------")
            
            frames = []
            while True:
                time.sleep(10)

                file_id, status = self.query_video_generation(task_id)
                if file_id != "":
                    frames = self.fetch_video_result(file_id)
                    print("---------------Successful---------------")
                    break
                elif status == "Fail" or status == "Unknown":
                    print(f"---------------Failed (status={status})--------")
                    break
            if frames: 
                return frames
            else:
                print(f"[Warning] No frames returned. Retrying ({attempt}/{max_retries})...")
        print(f"[Error] Failed to generate video after {max_retries} attempts.")        
        return []