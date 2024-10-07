import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import sys
import os

def scanner_video(rootDir):
    video_list = []
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            if file.endswith('.mp4'):
                video_list.append(file)
    return video_list

def detect_face(frame, face_detector,thresh=0.8):
    if frame is not None:
        h, w = frame.shape[:2]
        faces, kpss = face_detector.detect(frame,thresh=thresh)
        if len(kpss) > 0:
            return True               
    return False

def process_video(input_video_path, output_video_path, app):
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

        faces = app.get(frame)
        if len(faces) > 0 and faces[0]['det_score'] > 0.8:
            out.write(frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    video_path = sys.argv[1]
    save_path = sys.argv[2]
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640,640))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    videos = scanner_video(video_path)
    num = 0
    for video in videos:
        video_name = os.path.join(video_path, video)
        output = os.path.join(save_path, video)
        num += 1
        print(num)
        if os.path.exists(output):
            continue
        process_video(video_name, output, app)