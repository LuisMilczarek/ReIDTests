#!/usr/bin/env python

import os

import torchreid
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes

REID_MODEL_PATH = "./model.pt"
INPUT_FILE = "./input.mp4"


def main() -> None:

    cap = cv.VideoCapture(INPUT_FILE)

    yolo_model = YOLO("yolov8n-pose.pt")
    frame_num = 0

    length_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if length_frames > 100000:
        length_frames = 100000
    frames = []


    for i in range(length_frames):
        print(f"\rLoadind frames: {i:04d}/{length_frames:04d}",end="")
        ret, frame = cap.read()
        frames.append(frame)
    print("")

    print(f"Shape: {len(frames)}x{len(frames[0])}x{len(frames[0][0])}")
    
    #print("Converting to numpy")
    #frames = np.array(frames)
    #print("Conversion done")

    print("Infering")
    print("Infering done")

    if not os.path.exists("./output"):
        os.mkdir("output")

    trackers = {}

    #results = yolo_model.track(frames, stream= True)
    for frame in frames:
        print(f"\rProcessing bounding box: {frame_num:04d}/{length_frames:04d}",end="")
        result : Results
        result = yolo_model.track(frame,persist=True, verbose=False)[0]

        
        # print(result)
        bbox = result.boxes.cpu()
        #print(len(bbox))
        if result.boxes.is_track:
            ids = result.boxes.id.int().cpu().tolist()
        else:
            continue
        for id, (x1,y1,x2,y2) in zip(ids,bbox.xyxy):
            path = f"./output/{id:03d}"
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            if not os.path.exists(path=path):
                os.mkdir(path)
            if id not in trackers.keys():
                trackers[id] = 0
            trackers[id] += 1
            cv.imwrite(f"{path}/{trackers[id]:03d}.png", frame[y1:y2,x1:x2])
            #print(f"./output/{frame_num}/{id:03d}.png")
        frame_num += 1
        #ret, frame = cap.read()

    return None

if __name__ == "__main__":
    main()