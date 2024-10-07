

import os 
import cv2 
import sys 
from mimicmotion.dwpose.util import draw_pose
from mimicmotion.dwpose.dwpose_detector import dwpose_detector as dwprocessor
import numpy as np
import json


def scanner_video(rootDir):
    video_list = []
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            name = os.path.join(root, file)
            if file.endswith('.mp4'):
                video_list.append(name)
    return video_list

def get_image_pose(ref_image, idx):
    """process image pose

    Args:
        ref_image (np.ndarray): reference image pixel value

    Returns:
        np.ndarray: pose visual image in RGB-mode
    """
    frame_pose_score = {}
    height, width, _ = ref_image.shape
    ref_pose = dwprocessor(ref_image)
    body = ref_pose['bodies']['candidate']
    body_score = ref_pose['bodies']['score']
    hands_score = ref_pose['hands_score']
    hands= ref_pose['hands']
    faces_score = ref_pose['faces_score']
    frame_pose_score['frame_idx'] = idx
    frame_pose_score['body'] = body.tolist()
    frame_pose_score['hands'] = hands.tolist()
    frame_pose_score['body_score'] = body_score.tolist()
    frame_pose_score['hands_score'] = hands_score.tolist()
    frame_pose_score['faces_score'] = faces_score.tolist()
    return frame_pose_score
    

def process_video(input_video_path):
    video_json = {}
    video_name = input_video_path.split('/')[-1]
    video_json['video_name'] = video_name
    
    video_json['pose_score'] = []
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_json['total_frames'] =total_frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 姿态检测
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (384, 576))
        frame_pose_score = get_image_pose(frame_rgb, idx)
        video_json['pose_score'].append(frame_pose_score)
        idx += 1
    cap.release()
    return video_json
    
if __name__ == '__main__':
    video_path = sys.argv[1]
    save_path = sys.argv[2]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    videos = scanner_video(video_path)
    num = 1
    for video in videos:
        name = video.split('/')[-1].split('.mp4')[0] + '.json'
        save_name = os.path.join(save_path, name)
        if os.path.exists(save_name):
            continue
        video_json = process_video(video)
        print(num)
        with open(save_name, 'w') as json_file:
            json.dump(video_json, json_file)
        num +=1 
    
        


