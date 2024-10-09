#!/usr/bin/env python
import torchreid

from ultralytics import YOLO

REID_MODEL_PATH = "./model.pt"
INPUT_FILE = "./input.mp4"


def main() -> None:

    yolo_model = YOLO("yolov8n-pose.pt")

    return None

if __name__ == "__main__":
    main()