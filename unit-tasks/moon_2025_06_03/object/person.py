import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

import cv2
from ultralytics import YOLO


class YoloPersonDetector(Node):
    def __init__(self):
        super().__init__('yolo_person_detector')

        # 퍼블리셔: 사람 수를 퍼블리시
        self.publisher_ = self.create_publisher(Int32, 'person_count', 10)

        # YOLO 모델 로드
        self.model = YOLO('yolov8n.pt')  # 가장 가벼운 모델 사용

        # 웹캠 열기
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("웹캠을 열 수 없습니다.")
            exit()

        # 타이머: 0.1초마다 프레임 처리 (10Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("YOLO 사람 감지 노드 시작됨.")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("프레임을 읽을 수 없습니다.")
            return

        results = self.model(frame, verbose=False)
        boxes = results[0].boxes
        names = results[0].names

        person_count = 0
        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]
            if label == "person":
                person_count += 1

        msg = Int32()
        msg.data = person_count
        self.publisher_.publish(msg)
        self.get_logger().info(f"🙋‍♀️ 사람 수: {person_count}")

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloPersonDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()