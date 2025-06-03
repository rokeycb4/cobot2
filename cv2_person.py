import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import cv2
from ultralytics import YOLO

class YoloPersonCounter(Node):
    def __init__(self):
        super().__init__('yolo_person_counter')

        self.publisher_ = self.create_publisher(Int32, 'person_count', 10)
        self.model = YOLO('yolov8n.pt')
        self.cap = cv2.VideoCapture(2)

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("YOLO person counter started.")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to grab frame.")
            return

        results = self.model(frame)[0]
        count = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if self.model.names[cls_id] == 'person':
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 퍼블리시
        msg = Int32()
        msg.data = count
        self.publisher_.publish(msg)
        self.get_logger().info(f"Person count: {count}")

        # 화면 출력
        cv2.putText(frame, f'Person count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("YOLOv8 - Person Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = YoloPersonCounter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
