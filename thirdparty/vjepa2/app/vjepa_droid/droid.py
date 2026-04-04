# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
from logging import getLogger
from math import ceil

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu
import json
from transformers import AutoTokenizer

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=5,
    crop_size=224,
    rank=0,
    world_size=1,
    camera_views=0,
    stereo_view=False,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,
    tubelet_size=2,
):
    dataset = DROIDVideoDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        camera_views=camera_views,
        frameskip=tubelet_size,
        camera_frame=camera_frame,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info("VideoDataset unsupervised data loader created")

    return data_loader, dist_sampler


def get_json(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {filename}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {filename}: {e}")


class DROIDVideoDataset(torch.utils.data.Dataset):
    """Video classification dataset."""

    def __init__(
        self,
        data_path,
        camera_views=["left_mp4_path", "right_mp4_path"],
        frameskip=2,
        frames_per_clip=16,
        fps=5,
        transform=None,
        camera_frame=False,
    ):
        self.data_path = data_path
        self.frames_per_clip = frames_per_clip
        self.frameskip = frameskip
        self.fps = fps
        self.transform = transform
        self.camera_frame = camera_frame
        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Camera views
        # ---
        # wrist camera view
        # left camera view
        # right camera view
        self.camera_views = camera_views
        self.h5_name = "trajectory.h5"

        # Read CSV data
        self.df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.df)} samples from CSV: {data_path}")
        
        # for fair comparison, we use the same text encoder as WAN
        self.tokenizer = AutoTokenizer.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="tokenizer")
        
    def __getitem__(self, index):
        loaded_sample = self.df.iloc[index]

        # -- keep trying to load videos until you find a valid sample
        loaded_video = False
        while not loaded_video:
            try:
                buffer, indices, text_input_ids, mask = self.loadvideo_decord(loaded_sample)
                loaded_video = True
            except Exception as e:
                logger.info(f"Encountered exception when loading video {loaded_sample=} {e=}")
                loaded_video = False
                index = np.random.randint(self.__len__())
                loaded_sample = self.df.iloc[index]

        return buffer, indices, text_input_ids, mask

    def loadvideo_decord(self, loaded_sample):
        # -- load metadata from CSV row
        text_instruction = loaded_sample["action_text"]
        text_instruction = self.tokenizer(text_instruction, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
        text_input_ids, mask = text_instruction.input_ids[0], text_instruction.attention_mask[0]
        vpath = loaded_sample["video_path"]
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        # --
        fpc = self.frames_per_clip
        vlen = len(vr)

        if vlen < fpc:
            raise Exception(f"Video is too short {vpath=}, {fpc=}, {vlen=}")

        # sample a random window of consecutive frames
        sf = np.random.randint(0, vlen - fpc + 1)
        indices = np.arange(sf, sf + fpc).astype(np.int64)
        # --
        vr.seek(0)  # go to start of video before sampling frames
        buffer = vr.get_batch(indices).asnumpy()
        if self.transform is not None:
            buffer = self.transform(buffer)

        return buffer, indices, text_input_ids, mask

    def __len__(self):
        return len(self.df)
