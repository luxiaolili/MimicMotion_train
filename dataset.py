import json 
import os
import pickle
from pathlib import Path
from typing import Tuple, Literal

import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from transformers import CLIPImageProcessor, AutoTokenizer
import torch.nn.functional as F
from glob import glob
from mimicmotion.modules.pose_net import PoseNet

class TikTokDataset(data.Dataset):
    def __init__(self,
        video_folder,
        width=576,
        height=1024,
        sample_frames=25,
        sample_rate=4,
        bbox_crop=False,
        select_face=True,
        ):
        super(TikTokDataset, self).__init__()
        self.video_folder = video_folder
        self.width = width,
        self.height = height,
        self.sample_frames = sample_frames
        self.sample_rate = sample_rate
        self.bbox_crop = bbox_crop
        self.select_face = select_face,
        self.clip_image_procesor = CLIPImageProcessor()
        self.videos = os.listdir(self.video_folder)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

     
    def __len__(self):
        return len(self.videos)
    
    def set_clip_idx(self, video_length):
        clip_length = min(video_length, (self.sample_frames - 1) * self.sample_rate + 1)
        start_idx = random.randint(0, video_length - clip_length)
        clip_idxes = np.linspace(
            start_idx, start_idx + clip_length - 1, self.sample_frames, dtype=int
        ).tolist()
        return clip_idxes
    
    def __getitem__(self, index):
        video_name = self.videos[index]
        base_path = os.path.join(self.video_folder, video_name)
        img_path = os.path.join(base_path, 'imgs')
        face_path = os.path.join(base_path, 'faces')
        landmark_path = os.path.join(base_path, 'lmks')
        img_path_lst = sorted([img for img in glob(img_path +"/*.png")])
        lmk_path_lst = sorted([img for img in glob(landmark_path +"/*.png")])
        #reference image 
        if self.select_face:
            face_img_lst = sorted([img for img in glob(face_path+ "/*.png")])
            ref_img_name = random.choice(face_img_lst)
            
        else:
            ref_img_name = random.choice(img_path_lst)
        ref_img_pil = Image.open(ref_img_name)

        tgt_vidpil_lst = []
        tgt_vidlmk_lst = []

        #tgt frame and tgt landmark
        video_length = len(img_path_lst)
        clip_idxes = self.set_clip_idx(video_length)

        for c_idx in clip_idxes:
            tgt_img_path = img_path_lst[c_idx]
            
            tgt_img_pil = self.transform(Image.open(tgt_img_path).resize((self.width[0], self.height[0])))
            tgt_vidpil_lst.append(tgt_img_pil)

            tgt_lmk_path = lmk_path_lst[c_idx]
            tgt_lmk_pil = self.transform(Image.open(tgt_lmk_path).resize((self.width[0], self.height[0])))
            tgt_vidlmk_lst.append(tgt_lmk_pil)
        
        clip_img = self.clip_image_procesor(images=ref_img_pil, return_tensor="pt").pixel_values[0]
        ref_img = self.transform(ref_img_pil.resize((self.width[0], self.height[0])))
        tgt_vid = torch.stack(tgt_vidpil_lst, dim=0)
        tgt_lmk = torch.stack(tgt_vidpil_lst, dim=0)
    
        sample = {}
        sample['tgt_vid'] = tgt_vid
        sample['tgt_lmk'] = tgt_lmk
        sample['clip_img'] = clip_img
        sample['ref_img'] = ref_img
        return sample

def collate_fn(data):
    tgt_vid = torch.stack([example["tgt_vid"] for example in data])
    tgt_lmk = torch.stack([example["tgt_lmk"] for example in data])
    clip_img = torch.stack([torch.Tensor(example["clip_img"]) for example in data])
    ref_img = torch.stack([example["ref_img"] for example in data])

    return {
        "tgt_vid": tgt_vid,
        "tgt_lmk": tgt_lmk,
        "clip_img": clip_img,
        "ref_img": ref_img,
    }  

if __name__ == '__main__':
    dataroot_path = 'video_data'
    dataset = TikTokDataset(dataroot_path)
    posenet = PoseNet()
    train_dataloader = torch.utils.data.DataLoader(dataset,collate_fn=collate_fn,shuffle=True,batch_size=2,num_workers=4)
    for step, sample in enumerate(train_dataloader):
        print(sample['tgt_vid'].shape, sample['tgt_lmk'].shape, sample['clip_img'].shape, sample['ref_img'].shape)
        pose_latent = posenet(sample['tgt_lmk'])
        print(pose_latent.shape)
        break