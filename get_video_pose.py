import cv2
import numpy as np
from mimicmotion.dwpose.dwpose_detector import dwpose_detector as dwprocessor
from mimicmotion.dwpose.util import draw_pose
import sys
import os

def scanner_video(rootDir):
    video_list = []
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            if file.endswith('.mp4'):
                video_list.append(file)
    return video_list

def get_image_pose(ref_image):
    height, width, _ = ref_image.shape
    ref_pose = dwprocessor(ref_image)
    pose_img = draw_pose(ref_pose, height, width)
    return np.array(pose_img)

def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 姿态检测
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (384, 576))
        pose_result = get_image_pose(frame_rgb)
        pose_result = pose_result.transpose(1,2,0)
        pose_result = cv2.resize(pose_result, (width, height))
        # 写入输出视频
        out.write(pose_result)

    cap.release()
    out.release()

if __name__ == '__main__':
    video_path = sys.argv[1]
    save_path = sys.argv[2]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    videos = scanner_video(video_path)
    for video in videos:
        video_name = os.path.join(video_path, video)
        output = os.path.join(save_path, video)
        if os.path.exists(output):
            continue
        process_video(video_name, output)