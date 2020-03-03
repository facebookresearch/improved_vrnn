#
# Copyright (c) Facebook, Inc. and its affiliates.
#
"""
    cityscapes_laoder.py
    ~~~~~~~~~~~~~~

    Data loader for the Cityscapes dataset.
"""
import os
from glob import glob
import warnings

import parse
import numpy as np
import cv2
from torch.utils.data import Dataset

from conf import CITYSCAPES_PATH


FNAME_PATTERN = parse.compile('{frame_id:d}.jpg')


class CityscapesDataset(Dataset):

    def __init__(self, split, seq_len, normalize=True, img_side=64, resize=True, dataset_dir=CITYSCAPES_PATH):

        self.seq_len = seq_len
        self.img_side = img_side
        self.resize = resize
        data_dir = os.path.join(dataset_dir, split)
        self.example_dirs = sorted(glob(os.path.join(data_dir, '*', '*')))
        self.normalize = normalize
        self.split = split

    def __len__(self):
        if self.split.startswith('train'):
            return 45000
        else:
            return len(self.example_dirs)
        
    def _load_frame(self, frame_path):

        # Load image
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to desired image shape
        if self.resize:
            img = cv2.resize(img, (self.img_side, self.img_side))

        # Transpose channels to pytorch format and normalize 
        img = img.transpose((2, 0, 1))

        # Normalize the pixel intensities
        if self.normalize:
            img = img/255.0

        return img
        
    def __getitem__(self, item_idx):
        example_dir = self.example_dirs[item_idx % len(self.example_dirs)]

        # List jpgs in dir
        frames = glob(os.path.join(example_dir, '*.jpg'))

        # Parse frame filenames
        frames = [[f, FNAME_PATTERN.parse(os.path.basename(f)).named['frame_id']] 
                  for f in frames]

        # Sort by frame idx
        frames = sorted(frames, key=lambda x: x[1])

        # Return frames
        start_frame = 0
        if self.seq_len is not None:
            if self.split.startswith('train'):
                start_frame = np.random.randint(0, len(frames) - self.seq_len)
            else:
                start_frame = 0

        frames = frames[start_frame: start_frame + self.seq_len]

        frames = [self._load_frame(f[0]) for f in frames]
        frames = np.array(frames).astype(np.float32)

        return frames, frames
