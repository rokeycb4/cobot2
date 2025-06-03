import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import DR_init
import time

# ===== [고정: 로봇 초기화] =====
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
rclpy.init()
dsr_node = rclpy.create_node("dsr_control_node", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import amovej, wait, mwait, get_current_posj
except ImportError as e:
    print(f"[ERROR] DSR_ROBOT2 import failed: {e}")
    exit(1)

home_h = [0, 0, 90, -90, 90, 0]
home_a = [0, 0, 90, 0, 90, 0]


class DSRControl(Node):
    def __init__(self):
        super().__init__('dsr_control_node_logic')
        self.person_count = 0
        self.motion_paused = False
        self.toggle = True
        self.last_command_time = time.time()

        self.subscription = self.create_subscription(
            Int32,
            'person_count',
            self.person_count_callback,
            10
        )

    def person_count_callback(self, msg):
        self.person_count = msg.data


def main(args=None):
    node = DSRControl()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.time()

            # ===== 감지 인원 2명 이상 → 일시정지 + 현 위치로 고정 =====
            if node.person_count > 1:
                if not node.motion_paused:
                    node.get_logger().warn("⚠️ 2명 이상 감지! 로봇 일시정지")
                    node.motion_paused = True
                    mwait()
                    node.last_command_time = now  # 초기화

                # 1초에 한 번만 현 위치 이동 명령
                if now - node.last_command_time > 1.0:
                    current_pos = get_current_posj()
                    node.get_logger().info(f"[정지 유지] 현재 위치 재지정: {current_pos}")
                    amovej(current_pos, vel=20, acc=20)
                    node.last_command_time = now

            # ===== 감지 인원 1명 이하 → 정상 루틴 복귀 =====
            else:
                if node.motion_paused:
                    node.get_logger().info("✅ 인원 정상! 루틴 복귀")
                    node.motion_paused = False
                    node.last_command_time = now

                # 3초 간격으로 home 위치 번갈아 이동
                if now - node.last_command_time > 3.0:
                    target = home_h if node.toggle else home_a
                    node.get_logger().info(f"[이동] 위치: {target}")
                    amovej(target, vel=60, acc=60)
                    wait(2)  # 로봇이 이동할 시간 확보
                    node.toggle = not node.toggle
                    node.last_command_time = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
