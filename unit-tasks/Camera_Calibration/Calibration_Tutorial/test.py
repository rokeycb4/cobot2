import cv2
import rclpy
from rclpy.node import Node
from realsense import ImgNode
from scipy.spatial.transform import Rotation
from onrobot import RG

import time
import numpy as np


import DR_init

# for single robot
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"


# 마우스 콜백 함수
class TestNode(Node):
    def __init__(self):
        super().__init__("test_node")

        self.img_node = ImgNode()
        rclpy.spin_once(self.img_node)
        time.sleep(1)
        self.intrinsics = self.img_node.get_camera_intrinsic()
        self.gripper2cam = np.load("T_gripper2camera.npy")
        self.JReady = posj([0, 0, 90, 0, 90, -90])
        self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            depth_frame = self.img_node.get_depth_frame()
            while depth_frame is None or np.all(depth_frame == 0):
                self.get_logger().info("retry get depth img")
                rclpy.spin_once(self.img_node)
                depth_frame = self.img_node.get_depth_frame()

            print(f"img cordinate: ({x}, {y})")
            z = self.get_depth_value(x, y, depth_frame)
            camera_center_pos = self.get_camera_pos(x, y, z, self.intrinsics)
            print(f"camera cordinate: ({camera_center_pos})")

            gripper_coordinate = self.transform_to_base(camera_center_pos)
            print(f"gripper cordinate: ({gripper_coordinate})")

            self.pick_and_drop(*gripper_coordinate)
            print("=" * 100)

    def get_camera_pos(self, center_x, center_y, center_z, intrinsics):
        camera_x = (center_x - intrinsics["ppx"]) * center_z / intrinsics["fx"]
        camera_y = (center_y - intrinsics["ppy"]) * center_z / intrinsics["fy"]
        camera_z = center_z

        return (camera_x, camera_y, camera_z)

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def pick_and_drop(self, x, y, z):
        current_pos = get_current_posx()[0]
        pick_pos = posx([x, y, z, current_pos[3], current_pos[4], current_pos[5]])
        # TODO: Write pick and drop function
        #current_pos[2] -= 30
        pick_pos[2] -= 5
        movel(pick_pos,30,30)
        self.gripper.close_gripper()
        wait(1)
        h = posj(0,0,90,0,90,0)
        movej(h,30,30)
        # movej(...)  # move to initial position
        self.gripper.open_gripper()
        wait(1)
        

    def transform_to_base(self, camera_coords):
        """
        Converts 3D coordinates from the camera coordinate system
        to the robot's base coordinate system.
        """
        # gripper2cam = np.load(self.gripper2cam_path)
        coord = np.append(np.array(camera_coords), 1)  # Homogeneous coordinate

        base2gripper = self.get_robot_pose_matrix(*get_current_posx()[0])
        timer = time.time()

        base2cam = base2gripper @ self.gripper2cam
        td_coord = np.dot(base2cam, coord)

        return td_coord[:3]

    def open_img_node(self):
        rclpy.spin_once(self.img_node)
        img = self.img_node.get_color_frame()

        cv2.setMouseCallback("Webcam", self.mouse_callback, img)
        cv2.imshow("Webcam", img)

        # if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
        #     break

    def get_depth_value(self, center_x, center_y, depth_frame):
        height, width = depth_frame.shape
        if 0 <= center_x < width and 0 <= center_y < height:
            depth_value = depth_frame[center_y, center_x]
            return depth_value
        self.get_logger().warn(f"out of image range: {center_x}, {center_y}")
        return None


if __name__ == "__main__":
    rclpy.init()
    node = rclpy.create_node("dsr_example_demo_py", namespace=ROBOT_ID)

    DR_init.__dsr__node = node

    try:
        from DSR_ROBOT2 import (
            get_current_posx,
            movej,
            movel,
            wait,
        )

        from DR_common2 import posx, posj

    except ImportError as e:
        print(f"Error importing DSR_ROBOT2 : {e}")
        exit(True)
    # rclpy.init()

    cv2.namedWindow("Webcam")

    test_node = TestNode()
    # h = posj(0,0,90,0,90,0)
    # movej(h,30,30)
    while True:
        test_node.open_img_node()

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
            break

    cv2.destroyAllWindows()
