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
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32
import time
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
        amovel,
        amovej,
        amovesx,
        movesx,
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
        DR_FC_MOD_REL,
        DR_MV_MOD_ABS,
        drl_script_pause,
        drl_script_resume,
        set_robot_mode,
        DR_MVS_VEL_NONE,
        check_motion,
        set_velx,
        set_accx,
        change_operation_speed,


        DR_LINE,
        DR_BASE,
        posb,
        moveb

    )
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()
from DR_common2 import posx, posj

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

home_j = posj(0,0,90, 0,90,0)
camera_pos = posj(0,0,90,-90,90,0)

def trans_(org, tr):
    """org(posx)에 tr 오프셋 벡터 더해서 새로운 posx 반환"""
    return posx([o + t for o, t in zip(org, tr)])


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
        super().__init__("RobotController")
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
        self.boxSubscription = self.create_subscription(
            Float32MultiArray,
            'box_command',
            self.object_box_callback,
            10
        )
        self.faceSubscription = self.create_subscription(
            Float32MultiArray,
            'face_depth_command',
            self.face_callback,
            10
        )
        self.rotate_state = False # 물체 길이에 따라 회전 상태
        self.face_count = 0 #얼굴 탐지 카운트

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T
    
    #물체 길이 인식 후 회전 상태 반환
    def object_box_callback(self,msg):
        float_list = msg.data
        width = int(float_list[2] - float_list[0])  
        height = int(float_list[3] - float_list[1]) 
        self.rotate_state = width > height

    #얼굴 인식 callback
    def face_callback(self,msg):
        pos = msg.data.tolist()
        self.get_logger().info(f"[DEBUG] 수신된 얼굴 좌표 리스트: {pos}")
        gripper2cam_path = os.path.join(
            package_path, "resource", "T_gripper2camera.npy"
        )
        try:
            robot_posx = get_current_posx()[0]
            td_coord = self.transform_to_base(pos[:3], gripper2cam_path, robot_posx)
            robot_posx[3] += pos[3]
            robot_posx[4] += pos[4]
            if td_coord[2] and sum(td_coord) != 0:
                td_coord[2] += DEPTH_OFFSET - 10.0 
                td_coord[2] = max(td_coord[2], MIN_DEPTH)  
            target_pos = list(td_coord[:3]) + robot_posx[3:]
            target_pos[0] = min(max(target_pos[0], 200), 600)   
            target_pos[1] = min(max(target_pos[1], -170), -70)  
            target_pos[2] = min(max(target_pos[2], 400), 600)
            target_pos[3] = min(max(target_pos[3], 55), 105)      
            target_pos[4] = min(max(target_pos[4], -105), -75)
            if target_pos[0] < 250 and target_pos[2] < 450:
                target_pos[2] = 450
            self.face_count += 1
            if self.face_count > 30:
                self.send_message('안녕하세요.')
                self.face_count = 0
            self.get_logger().info(f"[DEBUG] 얼굴 좌표 -> 로봇 좌표 변환 결과 : {target_pos}")
            self.place_amove(target_pos,None,None,False)
        except IndexError as e:
            self.get_logger().info(f"[DEBUG] Error : {str(e)}")
            pass


    def transform_to_base(self, camera_coords, gripper2cam_path, robot_pos):
        gripper2cam = np.load(gripper2cam_path)
        coord = np.append(np.array(camera_coords), 1) 
        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)
        base2cam = base2gripper @ gripper2cam
        td_coord = np.dot(base2cam, coord)
        return td_coord[:3]

    def robot_control(self):
        get_keyword_future = self.get_keyword_client.call_async(self.get_keyword_request)
        self.get_logger().info(f'{get_keyword_future}')
        rclpy.spin_until_future_complete(self, get_keyword_future)
        get_keyword_result = get_keyword_future.result()

        if get_keyword_result.success:
            raw_message = get_keyword_result.message.strip()

            # 🔹 특수 명령어 처리
            if '홈' in raw_message:
                self.send_message('홈으로 이동합니다.')
                self.init_robot_with_camera(True)
                return
            if '이거' in raw_message:
                self.send_message('네')
                self.init_robot(True)

            # 🔹 포맷 정리
            if raw_message.startswith('[') and raw_message.endswith(']'):
                raw_message = raw_message[1:-1]
            if '/' not in raw_message:
                self.send_message('이해하지못했어요. 다시 말씀해주세요.')
                self.get_logger().error("포맷 오류: '/' 구분자가 없습니다. 받은 메시지: " + raw_message)
                return

            tool_part, location_part = raw_message.split('/')
            tool_list = tool_part.strip().split()
            location_list = location_part.strip().split()

            # 🔹 소통 응답 처리
            if tool_part.strip() == '소통':
                self.get_logger().error(f'{location_part}')
                if '(' in location_part and ')' in location_part:
                    location_part = location_part[1:-1]
                return self.send_message(location_part)


            # 🔹 기존 명령어 처리
            self.get_logger().info(f"도구 리스트: {tool_list}")
            self.get_logger().info(f"위치 리스트: {location_list}")

            for tool, location in zip(tool_list, location_list):
                tool = tool.lower()

                if 'left' in location or 'right' in location:
                    drop_dir = 'left' if 'left' in location else 'right'
                    self.send_message(f"{tool}를 {drop_dir}쪽에 옮깁니다.")
                    self.pick_and_placeb(tool, location)
                    continue  # ✅ 아래 가져와/가져가 로직으로 안 내려가게

                if '가져와' in location:
                    self.bring_tool_move(tool, location)
                elif '가져가' in location:
                    self.take_tool_move(tool, location)
                else:
                    self.send_message(f"알 수 없는 위치 지시어: {location}")


        else:
            message = get_keyword_result.message
            self.get_logger().warn(f"{message}")
            self.send_message(message)


    def get_target_pos(self, target):
        self.get_position_request.target = target
        self.get_logger().info("object_detection 노드로 깊이 위치 서비스 호출함")
        get_position_future = self.get_position_client.call_async(
            self.get_position_request
        )
        rclpy.spin_until_future_complete(self, get_position_future)

        if get_position_future.result():
            result = get_position_future.result().depth_position.tolist()
            self.get_logger().info(f"받은 깊이 위치: {result}")
            if sum(result) == 0:
                print("No target position")
                return None

            gripper2cam_path = os.path.join(
                package_path, "resource", "T_gripper2camera.npy"
            )
            while True:
                if check_motion() == 0:
                    break

            robot_posx = get_current_posx()[0]
            if target == '자동':
                td_coord = self.transform_to_base(result[:3], gripper2cam_path, robot_posx)
            if target == '가져와' or target =='가져가':
                td_coord = self.transform_to_base(result[:3], gripper2cam_path, robot_posx)
            else:
                td_coord = self.transform_to_base(result, gripper2cam_path, robot_posx)

            if td_coord[2] and sum(td_coord) != 0:
                td_coord[2] += DEPTH_OFFSET  
                td_coord[2] = max(td_coord[2], MIN_DEPTH)  
            target_pos = list(td_coord[:3]) + robot_posx[3:]
        return target_pos

    # 홈 위치
    def init_robot(self,griper_state=True):
        self.get_logger().info("init_robot")
        amovej(home_j,vel=VELOCITY, acc=ACC)
        self.get_logger().info("홈으로 이동중")
        if griper_state:
            self.gripper_open()
        mwait()

    #카메라 인식 위치
    def init_robot_with_camera(self,griper_state=True):
        self.get_logger().info("카메라 위치로 이동중")
        amovej(camera_pos, vel=VELOCITY, acc=ACC)
        if griper_state:
            self.gripper_open()
        mwait()
        self.get_logger().info("카메라 위치로 이동 완료")

    # 물체 잡아서 오른쪽/왼쪽에 놓기
    def pick_and_placeb(self, tool, drop_dir):
        """물체 집어서 오른쪽/왼쪽에 놓기"""
        print("[Tp_log] pick_and_place_target 실행")
        print("[Tp_log] drop_dir:", drop_dir)

        # 위치 초기화
        self.gripper_open()
        movej(home_j, vel=VELOCITY, acc=ACC)
        mwait()
        while True:
            if check_motion() == 0:
                break
        target_pos = self.get_target_pos(tool)
        if not target_pos:
            self.send_message(f"{tool}의 위치를 찾을 수 없어요.")
            return 

        # 공구 위치로 이동후 잡기
        movel(target_pos, vel=VELOCITY, acc=ACC)
        mwait()

        gripper.close_gripper()
        while gripper.get_status()[0]:
            time.sleep(0.5)

        cur_pos = get_current_posx()[0]

        # 오른쪽/왼쪽에 따라 drop 위치 결정
        pos_leftx, pos_lefty = 310, -270
        pos_rightx, pos_righty = 510, -270
        drop_xy = (pos_leftx, pos_lefty) if drop_dir == "left" else (pos_rightx, pos_righty)

        # 잡은 위치 기준으로 z축 +50 후, xy만 이동
        xy_moved_pos = posx([
            drop_xy[0], drop_xy[1], cur_pos[2] + 50,
            cur_pos[3], cur_pos[4], cur_pos[5]
        ])
        lowered_pos = trans_(xy_moved_pos, [0, 0, -90, 0, 0, 0])

        seg1 = posb(DR_LINE, xy_moved_pos, radius=20)
        seg2 = posb(DR_LINE, lowered_pos, radius=0)

        print("[Tp_log] moveb시작")
        moveb([seg1, seg2], vel=VELOCITY, acc=ACC, ref=DR_BASE, mod=0)

        print("[Tp_log] 힘제어 시작")
        force_control_on(-20)
        while not check_force_condition(DR_AXIS_Z, max=5):
            wait(0.1)
        force_control_off()

        gripper.open_gripper()
        while gripper.get_status()[0]:
            time.sleep(0.5)


        self.init_robot_with_camera()


    #물체로 집기
    def pick_and_place_target(self, target_pos):
        self.place_amove(target_pos,2,-5)
        self.gripper_close()
        self.get_logger().info("이동 완료")
    
    #물체 원래자리로 놓기
    def pick_and_place_drop(self,target_pos):
        self.place_move(target_pos,2,10)
        self.place_move(target_pos,2,-10)
        self.gripper_open()

    #해당위치로 동기 이동
    def place_move(self, target_pos , index = None , param = None):
        if index is not None and param is not None:
            target_pos[index] += param
        movel(target_pos, vel=VELOCITY, acc=ACC)

    #해당위치로 비동기 이동
    def place_amove(self, target_pos , index = None , param = None , w = True):
        if index is not None and param is not None:
            target_pos[index] += param
        amovel(target_pos, vel=VELOCITY, acc=ACC)
        self.current_pos = target_pos
        if w:
            while True:
                if check_motion() == 0:
                    break
    
    #물체 찾기
    def search_target(self):
        pos = get_current_posx()[0]
        self.place_amove(pos,0,50)

    #물체 가져다 놓기    
    def go_target(self,target):
        self.send_message(f'{target}을 가져다 놓을게요.')
        target_pos = self.target_positions[target]
        self.pick_and_place_drop(target_pos)
        self.init_robot_with_camera(True)

    #물체를 사람에게 전달
    def bring_tool_move(self,tool,position):
        hand_search_count = 0 # 어깨 위치 인식 카운트
        tool_search_count = 0 # 물체 위치 인식 카운트
        wait_count = 0  # 받을때 까지 기다리는 카운트
        get_state = True # 물건을 받았는지 확인 상태
        hand_pos = None
        while True:
            hand_pos = self.get_target_pos(position)
            if hand_pos is None or sum(hand_pos) == 0:
                hand_search_count += 1
            if hand_pos is not None or hand_search_count > 3000:
                break
        if hand_pos is None or sum(hand_pos) == 0:
            self.send_message('작업자 위치 확인이 안돼요')
            self.init_robot_with_camera(True)
            return
        hand_pos = self.safe_area(hand_pos)
        self.send_message('작업자 위치 확인했어요')
        self.init_robot()
        self.place_amove(get_current_posx()[0],0,-30,False)
        while True:
            target_pos = self.get_target_pos(tool)
            if tool_search_count > 5 or target_pos is not None:
                break
            if target_pos is None or sum(target_pos) == 0:
                self.search_target()
                tool_search_count += 1
        if target_pos is None or sum(target_pos) == 0:
            self.send_message(f'{tool}를 찾지 못했습니다.')
            self.init_robot_with_camera(False)
            return
        if self.rotate_state == True:
            target_pos[5] += 90
            self.rotate_state == False
        self.pick_and_place_target(target_pos)
        self.init_robot_with_camera(False)
        self.place_amove(hand_pos,1,300,False)
        self.send_message(f'{tool} 받으세요')
        while True:
            if check_motion() == 0:
                break
        force_control_on(1)
        while not check_force_condition(DR_AXIS_X, max=3):
            wait_count += 1
            if wait_count % 3000 == 0:
                self.send_message('팔 아파요')
            if wait_count > 10000:
                get_state = False
                break
            pass
        force_control_off()     
        if get_state:
            self.target_positions[tool] = target_pos
            self.gripper_open()
            self.send_message('감사합니다.')
        else:
            self.send_message('받지 않아서 원래 자리로 돌아가요.')
            self.init_robot_with_camera(False)
            self.pick_and_place_drop(target_pos)
        self.init_robot_with_camera(True)
    
    #물체를 제자리로
    def take_tool_move(self,tool,position):
        while True:
            hand_pos = self.get_target_pos(position)
            if hand_pos is not None and sum(hand_pos) != 0:
                break
        hand_pos = self.safe_area(hand_pos)
        self.place_move(hand_pos,1,200)
        self.send_message(f'{tool}을 주세요')
        force_control_on(1)
        while not check_force_condition(DR_AXIS_X, max=3) :
            pass
        force_control_off()
        self.gripper_close()
        self.send_message('감사합니다.')
        self.init_robot(False)
        self.go_target(tool)

    #TTS 퍼블리셔
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

    #로봇 이동영역 검사
    def safe_area(self,target_pos):
        target_pos[0] = min(max(target_pos[0], 0), 400)
        target_pos[1] = min(max(target_pos[1], -700), -400)
        target_pos[2] = min(max(target_pos[2], 200), 500)
        return target_pos
        

def main(args=None):
    node = RobotController()
    while rclpy.ok():
        node.robot_control()
    rclpy.shutdown()
    node.destroy_node()


if __name__ == "__main__":
    main()
