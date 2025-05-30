from ultralytics import YOLO
import os

model = YOLO("yolo11n.pt")
model.train(data="/home/rokey/Tutorial/ultralytics_ws/runs/detect/train6/weights/data.yaml", epochs=100, imgsz=640, batch=16)
