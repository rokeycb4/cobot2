import os
import time
import sys
from scipy.spatial.transform import Rotation
import numpy as np
import rclpy
from rclpy.node import Node
import DR_init
import json
from od_msg.srv import SrvDepthPosition
from std_msgs.msg import String
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory
from control.onrobot import RG

package_path = get_package_share_directory("dovis")

# for single robot
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60
# BUCKET_POS = [445.5, -242.6, 174.4, 156.4, 180.0, -112.5]
GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"
DEPTH_OFFSET = -5.0
MIN_DEPTH = 2.0

YOLO_MODEL_FILENAME = "best.pt"
YOLO_CLASS_NAME_JSON = "class_name_tool.json"

YOLO_JSON_PATH = os.path.join(package_path, "resource", YOLO_CLASS_NAME_JSON)

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
dsr_node = rclpy.create_node("robot_control_node", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import (
        movej, 
        movel, 
        get_current_posx, 
        mwait,
        wait,
        check_force_condition,
        DR_AXIS_X,
        DR_AXIS_Y,
        DR_AXIS_Z,
        task_compliance_ctrl,
        set_desired_force,
        release_force,
        release_compliance_ctrl,
        DR_FC_MOD_REL
    )
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()
from DR_common2 import posx, posj
gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

home_j = posj(0,0,90, 0,90,0)
camera_pos = posj(0,0,90,-90,90,0)

def force_control_on(force_z):
    """ z방향 힘제어"""
    k_d = [500.0,500.0,500.0,  1000,1000,1000]
    task_compliance_ctrl(k_d)
    wait(0.1)

    f_d = [0,0,force_z,0.0,0.0,0.0]
    f_dir = [0,0,1,0,0,0]
    set_desired_force(f_d,f_dir,mod=DR_FC_MOD_REL)

def force_control_off():
    """순응제어 해제, 힘제어 해제"""
    release_force()
    wait(0.5)
    release_compliance_ctrl()
    wait(0.1)

def compliance_control_on():
    """순응제어 켜기"""
    k_d = [500.0, 500.0, 500.0, 1000, 1000, 1000]
    task_compliance_ctrl(k_d)
    wait(0.1)

def compliance_control_off():
    release_compliance_ctrl()
    wait(0.1)

class RobotController(Node):
    def __init__(self):
        super().__init__("pick_and_place")
        self.get_logger().info("포지션 서비스 등록 중 ....")
        self.get_position_client = self.create_client(
            SrvDepthPosition, "/get_3d_position"
        )
        while not self.get_position_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("포지션 서비스 등록 대기중 ....")
        self.get_position_request = SrvDepthPosition.Request()
        self.get_logger().info("포지션 서비스 완료 ....")
        self.get_logger().info("키워드 서비스 등록중 ....")
        self.get_keyword_client = self.create_client(Trigger, "/get_keyword")
        while not self.get_keyword_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("키워드 서비스 등록 대기중 ....")
        self.get_keyword_request = Trigger.Request()
        self.get_logger().info("키워드 서비스 완료 ....")
        self.command_publisher = self.create_publisher(String, 'gpt_command', 10)
        with open(YOLO_JSON_PATH, "r", encoding="utf-8") as file:
            class_dict = json.load(file)
            self.items = list(class_dict.items())
        self.init_robot_with_camera()
        self.target_positions = {}
    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords, gripper2cam_path, robot_pos):
        gripper2cam = np.load(gripper2cam_path)
        coord = np.append(np.array(camera_coords), 1) 
        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)
        base2cam = base2gripper @ gripper2cam
        td_coord = np.dot(base2cam, coord)
        return td_coord[:3]

    def robot_control(self):
        target = None
        target_list = []
        get_keyword_future = self.get_keyword_client.call_async(self.get_keyword_request)
        self.get_logger().info(f'{get_keyword_future}')
        rclpy.spin_until_future_complete(self, get_keyword_future)
        get_keyword_result = get_keyword_future.result()
        if get_keyword_future.result().success:
            target_list = get_keyword_result.message.split()
            self.get_logger().info(f"키워드 서비스 응답: {target_list}")
            for target in target_list:
                if target == "춤":
                    self.get_logger().info("춤 명령을 받았습니다! 춤 동작을 시작합니다.")
                    self.perform_dance()
                    continue
                elif target == '홈' or target == 'home':
                    self.get_logger().info("홈으로 이동합니다.")
                    self.init_robot()
                    continue
                elif target == "자동":
                    max_attempts = 3000
                    attempt = 0
                    while attempt < max_attempts:
                        target_pos = self.get_target_pos(target)
                        if target_pos is None or sum(target_pos) == 0:
                            attempt += 1
                            continue
                        target_pos[2] = 450
                        self.place_move(target_pos)
                elif target == '망치':
                    hand_pos = self.get_target_pos(target)
                    if hand_pos is None:
                        self.send_message('위치 확인이 안돼요')
                        self.init_robot_with_camera(True)
                    else:
                        self.send_message('위치 확인했어요')
                        self.init_robot()
                        search_count = 0
                        while True:
                            target_pos = self.get_target_pos('Screwdriver')
                            if search_count > 5 or target_pos is not None:
                                break
                            if target_pos is None or sum(target_pos) == 0:
                                self.search_target()
                                search_count += 1
                        self.pick_and_place_target(target_pos)
                        self.init_robot_with_camera(False)
                        self.place_move(hand_pos,1,300)
                        self.send_message('받으세요')
                        count = 0
                        get_state = True
                        force_control_on(1)
                        while not check_force_condition(DR_AXIS_Z, max=3) or not check_force_condition(DR_AXIS_X, max=2) or not check_force_condition(DR_AXIS_Y, max=3):
                            count += 1
                            if count % 3000 == 0:
                                self.send_message('팔 아파요')
                            if count > 10000:
                                get_state = False
                                break
                            pass
                        force_control_off()
                        if get_state:
                            self.target_positions['Screwdriver'] = target_pos
                            self.gripper_open()
                            self.send_message('감사합니다.')
                            self.init_robot_with_camera(True)
                        else:
                            self.send_message('받지 않아서 원래 자리로 돌아가요.')
                            self.init_robot_with_camera(False)
                            self.pick_and_place_drop(target_pos)
                elif target == '가져가':
                    while True:
                        hand_pos = self.get_target_pos(target)
                        if hand_pos is not None and sum(hand_pos) != 0:
                            break
                    self.place_move(hand_pos,1,200)
                    while True:
                        target_pos = self.get_target_pos('Screwdriver')
                        if target_pos is not None and sum(target_pos) !=0:
                            self.send_message(f'Screwdriver 인식 확인했습니다.')
                            break
                    self.send_message('물건을 주세요')
                    force_control_on(1)
                    while not check_force_condition(DR_AXIS_Y, max=2):
                        pass
                    force_control_off()
                    self.gripper_close()
                    self.send_message('감사합니다.')
                    self.init_robot(False)
                    self.go_target('Screwdriver')
                else:
                    target_pos = self.get_target_pos(target)
                    if target_pos is None:
                        self.get_logger().warn("No target position")
                    else:
                        self.get_logger().info(f"target position: {target_pos}")
                        self.pick_and_place_target(target_pos)
        else:
            self.get_logger().warn(f"{get_keyword_result.message}")
            return

    def get_target_pos(self, target):
        self.get_position_request.target = target
        self.get_logger().info("call depth position service with object_detection node")
        get_position_future = self.get_position_client.call_async(
            self.get_position_request
        )
        rclpy.spin_until_future_complete(self, get_position_future)

        if get_position_future.result():
            result = get_position_future.result().depth_position.tolist()
            self.get_logger().info(f"Received depth position: {result}")
            if sum(result) == 0:
                print("No target position")
                return None

            gripper2cam_path = os.path.join(
                package_path, "resource", "T_gripper2camera.npy"
            )
            robot_posx = get_current_posx()[0]
            if target == '자동':
                td_coord = self.transform_to_base(result[:3], gripper2cam_path, robot_posx)
                robot_posx[3] += result[3]
                robot_posx[4] += result[4]
            if target == '망치' or target =='가져가':
                td_coord = self.transform_to_base(result[:3], gripper2cam_path, robot_posx)
            else:
                td_coord = self.transform_to_base(result, gripper2cam_path, robot_posx)

            if td_coord[2] and sum(td_coord) != 0:
                td_coord[2] += DEPTH_OFFSET  
                td_coord[2] = max(td_coord[2], MIN_DEPTH)  
            target_pos = list(td_coord[:3]) + robot_posx[3:]
        return target_pos

    def init_robot(self,griper_state=True):
        self.get_logger().info("init_robot")
        movej(home_j,vel=VELOCITY, acc=ACC)
        self.get_logger().info("홈으로 이동중")
        if griper_state:
            self.gripper_open()
        mwait()

    def init_robot_with_camera(self,griper_state=True):
        self.get_logger().info("카메라 위치로 이동중")
        movej(camera_pos, vel=VELOCITY, acc=ACC)
        if griper_state:
            self.gripper_open()
        mwait()
        self.get_logger().info("카메라 위치로 이동 완료")

    def pick_and_place_target(self, target_pos):
        target_pos[2] -= 5 
        movel(target_pos, vel=VELOCITY, acc=ACC)
        mwait()
        self.gripper_close()
        self.get_logger().info("이동 완료")
    
    def pick_and_place_drop(self,target_pos):
        self.place_move(target_pos,2,10)
        self.place_move(target_pos,2,-10)
        self.gripper_open()

    def place_move(self, target_pos , index = None , param = None):
        if index is not None and param is not None:
            target_pos[index] += param
        movel(target_pos, vel=VELOCITY, acc=ACC)

    def search_target(self):
        pos = get_current_posx()[0]
        pos[0] += 50
        movel(pos, vel=VELOCITY, acc=ACC)

    def search_no_target(self):
        classCount = len(self.items)
        self.get_logger().info(f"클래스 수 {classCount}")
        for i in range(classCount):
            count = 1
            for i in self.items:
                target_pos = self.get_target_pos(str(i))
                if target_pos is None or sum(target_pos) == 0:
                    count+=1
            pos = get_current_posx()[0]
            if classCount <= count:
                self.place_move(pos,2,-100)
                self.gripper_open()
                self.init_robot_with_camera()
                break
            else:
                self.search_target()
        
    def go_target(self,target):
        self.send_message(f'{target}을 가져다 놓을게요.')
        target_pos = self.target_positions[target]
        self.pick_and_place_drop(target_pos)
        self.init_robot_with_camera(True)

    def perform_dance(self):
        dance = [posx(367.31, 17.07, 450.32, 88.13, 179.98, 88.03),posx(380.31, 7.07, 400.32, 88.13, 179.98, 88.03),
                posx(307.31, 7.07, 450.32, 88.13, 179.98, 88.03),posx(367.31, 17.07, 370.32, 70.13, 179.98, 88.03)]
        for i in dance:
            movel(i, vel=VELOCITY, acc=ACC)
        mwait()

    def send_message(self,text):
        msg = String()
        msg.data = text
        self.command_publisher.publish(msg)

    def gripper_open(self):
        gripper.open_gripper()
        while gripper.get_status()[0]:
            time.sleep(0.5)
        wait(0.1)

    def gripper_close(self):
        gripper.close_gripper()
        while gripper.get_status()[0]:
            time.sleep(0.5)
        wait(0.1)
        

def main(args=None):
    node = RobotController()
    while rclpy.ok():
        node.robot_control()
    rclpy.shutdown()
    node.destroy_node()


if __name__ == "__main__":
    main()
