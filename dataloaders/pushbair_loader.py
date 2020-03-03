#
# Copyright (c) Facebook, Inc. and its affiliates.
#
import os
from glob import glob
import warnings

import parse
import numpy as np
import cv2
from torch.utils.data import Dataset

from conf import BAIRPUSH_PATH


FNAME_PATTERN = parse.compile('{frame_id:d}.png')
DIR_PATTERN = parse.compile('traj_{start_id:d}_to_{:d}')


class PushDataset(Dataset):

    def __init__(
        self, split, seq_len, 
        img_side=64, dataset_dir=BAIRPUSH_PATH, data_augmentation=True, normalize=True):

        self.split = split
        self.seq_len = seq_len
        self.img_side = img_side
        self.normalize = normalize

        data_dir = os.path.join(dataset_dir, split)
        example_dirs = glob(os.path.join(data_dir, '*', '*'))

        examples = []
        for ex_dir in example_dirs:
            # import pdb; pdb.set_trace()
            start_id = DIR_PATTERN.parse(os.path.basename(os.path.dirname(ex_dir))).named['start_id']
            start_id = int(start_id)
            delta = int(os.path.basename(ex_dir))
            example_id = start_id + delta - 1
            examples.append([example_id, ex_dir])

        self.example_dirs = sorted(examples, key=lambda x: x[0])

        # NOTE
        # If you want to use data augmentation, please use the transforms
        # in https://github.com/hassony2/torch_videovision

        # if self.split == 'train' and data_augmentation:
        #     self.transforms = video_transforms.Compose([
        #         video_transforms.ColorAdjustment(hue=30, brig=10, cont=10),
        #         video_transforms.RandomHorizontalFlip(),
        #         video_transforms.Crop(h_delta=5, w_delta=1)
        #     ])
        # else:
        #     self.transforms = None

        self.transforms = None

    def __len__(self):
        return len(self.example_dirs)
        
    def _load_frame(self, frame_path):

        # Load image
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to desired image shape
        if self.img_side != 128:
            img = cv2.resize(img, (self.img_side, self.img_side))


        return img
        
    def __getitem__(self, item_idx):
        example_dir = self.example_dirs[item_idx]

        # List jpgs in dir
        frames = glob(os.path.join(example_dir[1], '*.png'))

        # Parse frame filenames
        frames = [[f, FNAME_PATTERN.parse(os.path.basename(f)).named['frame_id']] 
                  for f in frames]

        # Sort by frame idx
        frames = sorted(frames, key=lambda x: x[1])

        # Return frames
        if self.seq_len is not None:
            if self.split == 'train':
                start_frame = np.random.randint(0, len(frames) - self.seq_len)
            else:
                start_frame = 0
            frames = frames[start_frame: start_frame + self.seq_len]

        frames = [self._load_frame(f[0]) for f in frames]
        frames = np.array(frames).astype(np.float32)
        frames = frames/255.

        if self.transforms is not None:
            frames = self.transforms(frames)

        frames = frames.transpose((0, 3, 1, 2))

        return (frames, item_idx)
