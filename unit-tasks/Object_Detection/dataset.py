import os
import shutil
from pathlib import Path

# 통합 클래스 정의
unified_classes = ['hammer', 'pliers', 'screwdriver', 'wrench']

# Tools V1 원래 클래스 → 통합 클래스
tool_to_unified = {
    'Ball-peen hammer': 'hammer',
    'Claw hammer': 'hammer',
    'hammer': 'hammer',
    'Combination pliers': 'pliers',
    'Diagonal pliers': 'pliers',
    'Linesman pliers': 'pliers',
    'Locking pliers': 'pliers',
    'Needle nose pliers': 'pliers',
    'Nose pliers': 'pliers',
    'Slip joint pliers': 'pliers',
    'Tongue and groove pliers': 'pliers',
    'Phillips screwdriver': 'screwdriver',
    'Pozidriv screwdriver': 'screwdriver',
    'Precision screwdriver': 'screwdriver',
    'Screwdriver': 'screwdriver',
    'Spanner screwdriver': 'screwdriver',
    'Square screwdriver': 'screwdriver',
    'Star screwdriver': 'screwdriver',
    'Torx screwdriver': 'screwdriver',
    'Interchangeable screwdriver': 'screwdriver',
    'Adjustable wrench': 'wrench',
    'Allen wrench': 'wrench',
    'Box-End wrench': 'wrench',
    'Open-End wrench': 'wrench',
    'Pipe wrench': 'wrench',
    'Ratchering wrench': 'wrench',
    'Socket wrench': 'wrench',
    'Torx wrench': 'wrench',
}

# Tools V1 클래스 리스트 (YOLO 내 data.yaml 기준)
tools_v1_class_list = ['Adjustable wrench', 'Allen', 'Allen wrench', 'Ball-peen hammer', 'Box-End wrench', 'Claw hammer', 'Combination pliers', 'Combination wrench', 'Diagonal pliers', 'Interchangeable', 'Interchangeable screwdriver', 'Linesman pliers', 'Locking pliers', 'Needle nose pliers', 'Nose pliers', 'Open-End', 'Open-End wrench', 'Phillips screwdriver', 'Pipe wrench', 'Pozidriv screwdriver', 'Precision screwdriver', 'Ratchering wrench', 'Screwdriver', 'Slip joint pliers', 'Socket wrench', 'Spanner screwdriver', 'Square screwdriver', 'Star screwdriver', 'Tongue and groove pliers', 'Torx screwdriver', 'Torx wrench'
                       ]  # data.yaml의 names 리스트 내용을 그대로 넣어주세요
tools_id_to_unified_id = {}
for idx, name in enumerate(tools_v1_class_list):
    if name in tool_to_unified:
        tools_id_to_unified_id[idx] = unified_classes.index(tool_to_unified[name])

def remap_label_file(label_path, id_map):
    with open(label_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if int(parts[0]) in id_map:
            parts[0] = str(id_map[int(parts[0])])
            new_lines.append(" ".join(parts) + "\n")

    with open(label_path, "w") as f:
        f.writelines(new_lines)

# 병합 함수
def merge_yolo_dataset(tools_dir, screwdriver_dir, output_dir):
    for split in ['train', 'valid', 'test']:
        for dtype in ['images', 'labels']:
            tools_path = Path(tools_dir) / split / dtype
            screw_path = Path(screwdriver_dir) / split / dtype
            out_path = Path(output_dir) / split / dtype
            out_path.mkdir(parents=True, exist_ok=True)

            # tools 복사 및 라벨 재매핑
            if tools_path.exists():
                for file in tools_path.glob("*.*"):
                    dest = out_path / file.name
                    shutil.copy(file, dest)
                    if dtype == "labels":
                        remap_label_file(dest, tools_id_to_unified_id)

            # screwdriver 복사 (라벨 그대로)
            if screw_path.exists():
                for file in screw_path.glob("*.*"):
                    shutil.copy(file, out_path / file.name)

# 사용 예시
tools_dir = "/home/rokey/Downloads/Tools V1.v1i.yolov11"
screwdriver_dir = "/home/rokey/Downloads/screwdriver_subset.v2i.yolov11"
output_dir = "/home/rokey/ros2_ws/src/DoosanBootcamp3rd/dsr_rokey/pick_and_place_text/pick_and_place_text"

merge_yolo_dataset(tools_dir, screwdriver_dir, output_dir)

# import yaml

# data_yaml = {
#     'train': './train/images',
#     'val': './valid/images',
#     'test': './test/images',
#     'names': ['hammer', 'pliers', 'screwdriver', 'wrench']
# }

# with open('/home/rokey/ros2_ws/src/DoosanBootcamp3rd/dsr_rokey/pick_and_place_text/pick_and_place_text/data.yaml', 'w') as f:
#     yaml.dump(data_yaml, f)

# print("✅ data.yaml 생성 완료")


# from ultralytics import YOLO
# import cv2

# model = YOLO("/home/rokey/ros2_ws/src/DoosanBootcamp3rd/dsr_rokey/pick_and_place_text/resource/best.pt")

# img_path = "/home/rokey/Tutorial/ultralytics_ws/test3.jpg"  # 테스트 이미지 경로
# results = model.predict(source=img_path, conf=0.25, save=True)

# print("Detected class IDs:", results[0].boxes.cls)


from ultralytics import YOLO

model = YOLO("/home/rokey/ros2_ws/src/DoosanBootcamp3rd/dsr_rokey/pick_and_place_text/resource/best.pt")
results = model.predict(source="/home/rokey/Tutorial/ultralytics_ws/test4.jpg", conf=0.25, save=True)
results[0].show()  # bounding box 시각화
print(results[0].boxes.cls)  # 클래스 ID들 출력


