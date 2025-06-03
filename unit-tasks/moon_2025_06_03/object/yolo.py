########## YoloModel ##########
import os
import json
import time
from collections import Counter
import cv2
import mediapipe as mp

import rclpy
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO
import numpy as np

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
# mp_draw = mp.solutions.drawing_utils

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False)
hands = mp_hands.Hands(static_image_mode=False)
mp_draw = mp.solutions.drawing_utils

PACKAGE_NAME = "dovis"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

YOLO_MODEL_FILENAME = "yolo8_2.pt"
YOLO_CLASS_NAME_JSON = "class_name_tool2.json"

YOLO_MODEL_PATH = os.path.join(PACKAGE_PATH, "resource", YOLO_MODEL_FILENAME)
YOLO_JSON_PATH = os.path.join(PACKAGE_PATH, "resource", YOLO_CLASS_NAME_JSON)


class YoloModel:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)
        with open(YOLO_JSON_PATH, "r", encoding="utf-8") as file:
            class_dict = json.load(file)
            self.class_dict = class_dict
            self.reversed_class_dict = {v: int(k) for k, v in class_dict.items()}
            
    
    def get_class_dict(self):
        return self.class_dict.items()

    def get_frames(self, img_node, duration=1.0):
        """get frames while target_time"""
        end_time = time.time() + duration
        frames = {}

        while time.time() < end_time:
            rclpy.spin_once(img_node)
            frame = img_node.get_color_frame()
            stamp = img_node.get_color_frame_stamp()
            if frame is not None:
                frames[stamp] = frame
            time.sleep(0.01)

        if not frames:
            print("No frames captured in %.2f seconds", duration)

        print("%d frames captured", len(frames))
        return list(frames.values())

    def get_best_detection(self, img_node, target):
        rclpy.spin_once(img_node)
        frames = self.get_frames(img_node)
        if not frames:  # Check if frames are empty
            return None

        results = self.model(frames, verbose=False)
        print("classes: ")
        print(results[0].names)
        detections = self._aggregate_detections(results)
        try:
            label_id = self.reversed_class_dict[target]
            print("label_id: ", label_id)
        except KeyError:
            return None, None
        print("detections: ", detections)

        matches = [d for d in detections if d["label"] == label_id]
        if not matches:
            print("No matches found for the target label.")
            return None, None
        best_det = max(matches, key=lambda x: x["score"])
        return best_det["box"], best_det["score"]

    def _aggregate_detections(self, results, confidence_threshold=0.5, iou_threshold=0.5):
        """
        Fuse raw detection boxes across frames using IoU-based grouping
        and majority voting for robust final detections.
        """
        raw = []
        for res in results:
            for box, score, label in zip(
                res.boxes.xyxy.tolist(),
                res.boxes.conf.tolist(),
                res.boxes.cls.tolist(),
            ):
                if score >= confidence_threshold:
                    raw.append({"box": box, "score": score, "label": int(label)})

        final = []
        used = [False] * len(raw)

        for i, det in enumerate(raw):
            if used[i]:
                continue
            group = [det]
            used[i] = True
            for j, other in enumerate(raw):
                if not used[j] and other["label"] == det["label"]:
                    if self._iou(det["box"], other["box"]) >= iou_threshold:
                        group.append(other)
                        used[j] = True

            boxes = np.array([g["box"] for g in group])
            scores = np.array([g["score"] for g in group])
            labels = [g["label"] for g in group]

            final.append(
                {
                    "box": boxes.mean(axis=0).tolist(),
                    "score": float(scores.mean()),
                    "label": Counter(labels).most_common(1)[0][0],
                }
            )

        return final

    def _iou(self, box1, box2):
        """
        Compute Intersection over Union (IoU) between two boxes [x1, y1, x2, y2].
        """
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
    
    def get_shoulder_detection(self, img_node):
        rclpy.spin_once(img_node)
        frames = self.get_frames(img_node)
        if not frames:
            print("[DEBUG] 손 인식용 프레임이 없습니다.")
            return None
        latest_frame = frames[-1]
        hand_pos = self.detect_pose(latest_frame,mp_pose.PoseLandmark.RIGHT_SHOULDER)
        print(f"[DEBUG] 감지된 손 위치: {hand_pos}")
        return hand_pos
    
    def get_hand_detection(self, img_node):
        rclpy.spin_once(img_node)
        frames = self.get_frames(img_node)
        if not frames:
            print("[DEBUG] 손 인식용 프레임이 없습니다.")
            return None
        latest_frame = frames[-1]
        hand_pos = self.detect_pose(latest_frame,mp_pose.PoseLandmark.RIGHT_WRIST)
        print(f"[DEBUG] 감지된 손 위치: {hand_pos}")
        return hand_pos
    
    def get_face_detection(self,img_node):
        rclpy.spin_once(img_node)
        frames = self.get_frames(img_node)
        if not frames:
            print("[DEBUG] 손 인식용 프레임이 없습니다.")
            return None
        latest_frame = frames[-1]
        face_pos = self.detect_pose(latest_frame,mp_pose.PoseLandmark.NOSE)
        print(f"[DEBUG] 감지된 얼굴 위치: {face_pos}")
        return face_pos
    
    def get_hand_detection2(self,img_node):
        rclpy.spin_once(img_node)
        frames = self.get_frames(img_node)
        if not frames:
            print("[DEBUG] 손 인식용 프레임이 없습니다.")
            return None
        latest_frame = frames[-1]
        hand_pos = self.detect_hand2(latest_frame)
        print(f"[DEBUG] 감지된 손 위치: {hand_pos}")
        return hand_pos
        

    def detect_pose(self,frame,landmark):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        results = pose.process(img_rgb)
        if results.pose_landmarks:
            position = results.pose_landmarks.landmark[landmark]
            height, width, _ = frame.shape
            center_x = width // 2
            center_y = height // 2
            x_pixel = int(position.x * width)
            y_pixel = int(position.y * height)
            if abs(x_pixel - center_x) < 20:
                rx = 0
            elif x_pixel < center_x:
                rx = -3
            else:
                rx = 3
            if abs(y_pixel - center_y) < 20:
                ry = 0
            elif y_pixel < center_y:
                ry = -3
            else:
                ry = 3
            print(f"[DEBUG] detect 좌표: ({x_pixel}, {y_pixel})") 
            return (x_pixel, y_pixel,rx,ry)
        return None
    
    def detect_hand2(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        results = hands.process(img_rgb)
        if results.pose_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x_pixel = int(index_tip.x * width)
                y_pixel = int(index_tip.y * height)
                print(f"[DEBUG] 검지 끝 좌표: ({x_pixel}, {y_pixel})")
                return (x_pixel, y_pixel)
        return None
    
    # def detect_hand(self, frame):
    #     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     height, width, _ = frame.shape
    #     results = pose.process(img_rgb)
    #     if results.pose_landmarks:
    #         wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    #         height, width, _ = frame.shape
    #         center_x = width // 2
    #         center_y = height // 2
    #         x_pixel = int(wrist.x * width)
    #         y_pixel = int(wrist.y * height)
    #         if y_pixel < center_y:
    #             rx = 1   
    #         else:
    #             rx = -1  

    #         if x_pixel < center_x:
    #             ry = 1   
    #         else:
    #             ry = -1  
    #         print(f"[DEBUG] 검지 끝 좌표: ({x_pixel}, {y_pixel})")  # 🟢 디버깅용 출력
    #         return (x_pixel, y_pixel,rx,ry)
    #     return None
    
    # def detect_shoulder(self, frame):
    #     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     height, width, _ = frame.shape
    #     results = pose.process(img_rgb)
    #     if results.pose_landmarks:
    #         shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    #         height, width, _ = frame.shape
    #         center_x = width // 2
    #         center_y = height // 2
    #         x_pixel = int(shoulder.x * width)
    #         y_pixel = int(shoulder.y * height)
    #         if y_pixel < center_y:
    #             rx = 1   
    #         else:
    #             rx = -1  

    #         if x_pixel < center_x:
    #             ry = 1   
    #         else:
    #             ry = -1  
    #         print(f"[DEBUG] 검지 끝 좌표: ({x_pixel}, {y_pixel})")  # 🟢 디버깅용 출력
    #         return (x_pixel, y_pixel,rx,ry)
    #     return None
    