import numpy as np
import rclpy
from rclpy.node import Node
from typing import Any, Callable, Optional, Tuple

from ament_index_python.packages import get_package_share_directory
from od_msg.srv import SrvDepthPosition
from object.realsense import ImgNode
from object.yolo import YoloModel
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Bool

PACKAGE_NAME = 'dovis'
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)


class ObjectDetectionNode(Node):
    def __init__(self, model_name = 'yolo'):
        super().__init__('object_detection_node')
        self.img_node = ImgNode()
        self.model = self._load_model(model_name)
        self.intrinsics = self._wait_for_valid_data(
            self.img_node.get_camera_intrinsic, "camera intrinsics"
        )
        self.create_service(
            SrvDepthPosition,
            'get_3d_position',
            self.handle_get_depth
        )

        self.face_subscription = self.create_subscription(
            Bool,
            'face_command',
            self.face_callback,
            10
        )

        self.face_publisher = self.create_publisher(Float32MultiArray, 'face_depth_command', 10)

        self.box_publisher = self.create_publisher(Float32MultiArray, 'box_command', 10)
        self.get_logger().info("ObjectDetectionNode initialized.")

    def face_callback(self,msg):
        command = msg.data
        self.get_logger().info(f"수신된 명령: {command}")
        if command:
            coords = self._compute_position('얼굴')
            #coords = 0.0 , 0.0 , 0.0
            depth_position = [float(x) for x in coords]
            if sum(depth_position) != 0:
                msg = Float32MultiArray()
                msg.data = depth_position
                self.face_publisher.publish(msg)

    def _load_model(self, name):
        """모델 이름에 따라 인스턴스를 반환합니다."""
        if name.lower() == 'yolo':
            return YoloModel()
        raise ValueError(f"Unsupported model: {name}")

    def handle_get_depth(self, request, response):
        """클라이언트 요청을 처리해 3D 좌표를 반환합니다."""
        self.get_logger().info(f"Received request: {request}")
        coords = self._compute_position(request.target)
        response.depth_position = [float(x) for x in coords]
        return response
    
    def _compute_position(self, target):
        """이미지를 처리해 객체의 카메라 좌표를 계산합니다."""
        rclpy.spin_once(self.img_node)
        if target == '가져와' or target == '가져가' or target == '얼굴':
            if target == '가져가':
                hand_pos = self.model.get_hand_detection(self.img_node)
            elif target == '가져와':
                hand_pos = self.model.get_shoulder_detection(self.img_node)
            elif target == '이거':
                hand_pos = self.model.get_hand_detection2(self.img_node)
            elif target == '얼굴':
                hand_pos = self.model.get_face_detection(self.img_node)
            if hand_pos is not None:
                x, y , rx , ry  = hand_pos
                self.get_logger().info(f"x={x}")
                self.get_logger().info(f"y={y}")
                self.get_logger().info(f"rx={rx}")
                self.get_logger().info(f"ry={ry}")
                cz = self._get_depth(x, y)
                if cz is None:
                    self.get_logger().warn("깊이 값이 범위를 벗어났습니다.")
                    return 0.0, 0.0, 0.0
                return self._pixel_to_camera_coords(x, y, cz , rx , ry )
            else:
                self.get_logger().warn("손 검출 실패.")
                return 0.0, 0.0, 0.0
        box, score = self.model.get_best_detection(self.img_node, target)
        if box is None or score is None:
            self.get_logger().warn("객체 검출 실패.")
            return 0.0, 0.0, 0.0
        
        self.get_logger().info(f"Detection: box={box}, score={score}")
        cx, cy = map(int, [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        cz = self._get_depth(cx, cy)
        if cz is None:
            self.get_logger().warn("깊이 값이 범위를 벗어났습니다.")
            return 0.0, 0.0, 0.0
        msg = Float32MultiArray()
        msg.data = [box[i] for i in range(4)]
        self.box_publisher.publish(msg)
        return self._pixel_to_camera_coords(cx, cy, cz)

    def _get_depth(self, x, y):
        """픽셀 좌표의 depth 값을 안전하게 읽어옵니다."""
        frame = self._wait_for_valid_data(self.img_node.get_depth_frame, "depth frame")
        try:
            return frame[y, x]
        except IndexError:
            self.get_logger().warn(f"Coordinates ({x},{y}) out of range.")
            return None

    def _wait_for_valid_data(self, getter, description):
        """getter 함수가 유효한 데이터를 반환할 때까지 spin 하며 재시도합니다."""
        data = getter()
        while data is None or (isinstance(data, np.ndarray) and not data.any()):
            rclpy.spin_once(self.img_node)
            self.get_logger().info(f"Retry getting {description}.")
            data = getter()
        return data

    def _pixel_to_camera_coords(self, x, y, z , rx = None , ry = None):
        """픽셀 좌표와 intrinsics를 이용해 카메라 좌표계로 변환합니다."""
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        ppx = self.intrinsics['ppx']
        ppy = self.intrinsics['ppy']
        if rx is None and ry is None:
            return (
                (x - ppx) * z / fx,
                (y - ppy) * z / fy,
                z
            )
        else:
            return (
                (x - ppx) * z / fx,
                (y - ppy) * z / fy,
                z,
                rx,
                ry
            )


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
