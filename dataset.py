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
import torch.nn.functional as F
from pathlib import Path 
from decord import VideoReader
from PIL import Image, ImageDraw


def draw_mask(frame, x0, y0, x1, y1, score=1., margin=10):
    H, W = frame.shape[-2:]
    x0 = int(x0 * W)
    x1 = int(x1 * W)
    y0 = int(y0 * H)
    y1 = int(y1 * H)
    x0, y0 = max(x0 - margin, 0), max(y0 - margin, 0)
    x1, y1 = min(x1 + margin, W), min(y1 + margin, H)
    frame[..., y0:y1, x0:x1] = score
    return frame
    

def get_hands_box(hands_point):
    all_hands_point = []
    for i in range(hands_point.shape[0]):
        hand_points = hands_point[i,:]
        min_point = np.min(hand_points, axis=0) 
        max_point = np.max(hand_points, axis=0)  
        all_hands_point.append((min_point, max_point))
    return all_hands_point


def get_hands_mask(json_file, frame_ids, shape):
    height = 1024
    width = 576
    hands_mask = torch.zeros((shape, 1, height, width))
    
    try:
        with open(json_file, 'r') as json_data:
            video_json = json.load(json_data)
        video_name = video_json['video_name']
        pose_score = video_json['pose_score']
        frame_num = 0
        for idx in frame_ids:
            hand_boxes = np.array(pose_score[idx]['hands'])
            hands_scores = np.array(pose_score[idx]['hands_score'])
            hands_boxes = get_hands_box(hand_boxes)
            for hands_idx, ((x0, y0), (x1, y1)) in enumerate(hands_boxes):
                if hands_scores.mean() > 0.7:
                    hands_score = 0.8 # max((hands_score - 0.5), 0) / 0.5
                else:
                    hands_score = 0.
                
                frame = torch.zeros((1, height, width))
                frame_mask = draw_mask(frame, x0, y0, x1, y1, hands_score)
                
                hands_mask[frame_num] = frame_mask
            frame_num += 1
    except Exception as e:
        print(f"发生了一个错误：{e}")

    return hands_mask


class TikTokDataset(data.Dataset):
    def __init__(self,
        video_folder,
        width=576,
        height=768,
        sample_frames=14,
        sample_rate=3,
        is_image=False,
        ):
        super(TikTokDataset, self).__init__()
        self.video_folder = video_folder
        self.width = 576,
        self.height = 768,
        self.sample_frames = sample_frames
        self.sample_rate = sample_rate
        # self.videos = self.get_videos(self.video_folder)
        self.videos = self.get_imgs(self.video_folder)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize([height, width], antialias=True),
            transforms.Normalize([0.5], [0.5])
        ])
        self.is_image = is_image
        self.ref_transforms = transforms.Compose([
            
            transforms.Resize([height, width], antialias=True),
            transforms.ToTensor(),
        ])
        self.mask_transforms = transforms.Compose([
                transforms.Resize([height, width], antialias=True),
            ])

     
    def __len__(self):
        return len(self.videos)

    def get_videos(self, video_folder):
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm']
        path = Path(os.path.join(video_folder, 'videos'))
        video_files = [file.name for file in path.iterdir() if file.suffix in video_extensions]
        return video_files


    def get_imgs(self, imgs_folder):
        img_extensions = ['.png']
        videos = []
        path = Path(os.path.join(imgs_folder, 'ref'))
        img_files = [file.name for file in path.iterdir() if file.suffix in img_extensions]
        for img in img_files:
            video = img.split('.')[0] + '.mp4'
            videos.append(video)
        return videos

    def select_pose_frame(self, json_file):
        frame_list = []
        try:
            with open(json_file, 'r') as json_data:
                video_json = json.load(json_data)
            video_name = video_json['video_name']
            pose_score = video_json['pose_score']
            bad_list = []
            select_thresh = 2
            for idx in range(len(pose_score)):
                frame_idx = pose_score[idx]['frame_idx']
                body = np.array(pose_score[idx]['body'])
                hands = np.array(pose_score[idx]['hands'])
                body_score = np.array(pose_score[idx]['body_score'])
                hands_score = np.array(pose_score[idx]['hands_score'])
                faces_score = np.array(pose_score[idx]['faces_score'])
                body_score_mean = body_score.mean()
                hands_score_mean = hands_score.mean()
                faces_score_mean = faces_score.mean()
                num = 0
                if body_score_mean < 0.5:
                    num += 1
                if hands_score_mean < 0.5:
                    num += 1
                if faces_score_mean < 0.5:
                    num += 1
                if num >=2:
                    bad_list.append(video_name)
                else:
                    frame_list.append(frame_idx)
        except:
            print(json_file)
        return frame_list


    def get_batch(self, idx):
        video_name = self.videos[idx]
        video_dir = os.path.join(self.video_folder, 'videos', video_name)
        video_pose_dir = os.path.join(self.video_folder, 'dwpose', video_name)
        img_name = video_name.split('.')[0] + '.png'
        json_name = video_name.split('.')[0] + '.json'
        reference_img_name = os.path.join(self.video_folder, 'ref', img_name)
        pose_json_name = os.path.join(self.video_folder, 'pose_score', json_name)

        video_reader = VideoReader(video_dir)
        video_reader_pose = VideoReader(video_pose_dir)

        assert len(video_reader) == len(video_reader_pose), f"len(video_reader) != len(video_reader_pose) in video {idx}"
        

        frame_idx = self.select_pose_frame(pose_json_name)
        video_length = len(frame_idx)


        clip_length = min(video_length, (self.sample_frames - 1) * self.sample_rate + 1)
        start_idx   = random.randint(0, video_length - clip_length)
        np_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_frames, dtype=int).tolist()
        frame_index = [frame_idx[i] for i in np_index]
        if not self.is_image:
            batch_index = np.array(frame_index)
        else:
            batch_index = [random.choice(frame_idx)]
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader
        
        pixel_values_pose = torch.from_numpy(video_reader_pose.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values_pose = pixel_values_pose / 255.
        del video_reader_pose

        ref_img = Image.open(reference_img_name)
        if self.is_image:
            pixel_values = pixel_values[0]
            pixel_values_pose = pixel_values_pose[0]
       
        # hands_mask = get_hands_mask(pose_json_name, frame_index, pixel_values_pose.shape[0])
        return pixel_values, pixel_values_pose, ref_img

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, pose_pixel_values, reference_image = self.get_batch(idx)
                break
            except Exception as e:
                print(e)
                idx = random.randint(0, len(self.videos) - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        pose_pixel_values = self.pixel_transforms(pose_pixel_values)
        reference_image = self.ref_transforms(reference_image)
        # hands_mask = self.mask_transforms(hands_mask)
        sample = dict(
            pixel_values=pixel_values, 
            pose_pixel_values=pose_pixel_values,
            reference_image=reference_image,
            #hands_mask = hands_mask,
        )
        return sample

if __name__ == '__main__':
     video_folder = 'ubc_data'
     dataset = TikTokDataset(video_folder,is_image=False)
     dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=8)
     for idx, batch in enumerate(dataloader):
        print("pixel_values: ", idx, batch["pixel_values"].shape, batch['pose_pixel_values'].shape, batch['reference_image'].shape)
        break