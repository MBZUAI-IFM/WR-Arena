import torch
import torchvision
import os

def adjust_video_fps(input_path, output_path,target_fps=45):
    video = torchvision.io.read_video(input_path, pts_unit='sec')
    frames = video[0]

    torchvision.io.write_video(
        output_path,
        frames,
        fps=target_fps,
        video_codec='h264'  
    )

if __name__ == "__main__":
    input_video = "example.mp4"
    adjust_video_fps(input_video)