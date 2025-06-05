import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation

# 1) 로봇 그리퍼의 절대 좌표 (x, y, z, rx, ry, rz)를 행렬로 변환하는 함수
def get_robot_pose_matrix(x, y, z, rx, ry, rz):
    """
    베이스->그리퍼 변환행렬 (4x4)을 반환.
    """
    R = Rotation.from_euler('ZYZ', [rx, ry, rz], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


# 2) 체커보드 코너 검출 (카메라→체커보드 변환 구하기)
def find_checkerboard_pose(
    image, board_size, square_size, camera_matrix, dist_coeffs
):
    """
    checkerboard_size = (7, 5)  # 내부 코너 개수
    square_size = 25.0          # mm 단위
    이미지에서 체커보드를 찾고, solvePnP로 카메라→체커보드 변환(R, t)을 구함.
    반환값: (R_camera2checker, t_camera2checker)
    """
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    # 예: x 방향으로 square_size씩 증가, y 방향으로 square_size씩 증가
    objp[:, :2] = (
        np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2) * 25
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray,
        board_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None, None

    # 코너 좌표를 더 정확히
    corners_sub = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )

    # solvePnP
    retval, rvec, tvec = cv2.solvePnP(objp, corners_sub, camera_matrix, dist_coeffs)
    if not retval:
        return None, None

    # 회전벡터 -> 회전행렬
    R, _ = cv2.Rodrigues(rvec)

    return R, tvec


def calibrate_camera_from_chessboard(
    image_folder_path,
    board_size,  # (7, 5)처럼 내부 코너 개수
    square_size,  # mm 단위
):
    """
    지정된 폴더 안의 체커보드 이미지를 읽고, 카메라 행렬(camera_matrix)와 왜곡 계수(dist_coeffs)를 추정한다.
    board_size: 체커보드 내부 코너 수 (cols, rows)
    square_size: 체커보드 한 칸 크기 (mm)
    """
    # 3D 세계 좌표계에 대한 좌표 생성 (z=0 평면 상에 체커보드)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    # 예: x 방향으로 square_size씩 증가, y 방향으로 square_size씩 증가
    objp[:, :2] = (
        np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2) * square_size
    )

    # 모든 이미지에 대해 3D / 2D 포인트 누적
    obj_points = []  # 3D world points
    img_points = []  # 2D image points
    image_shape = None

    # 폴더 내에 있는 이미지 파일 읽기
    image_paths = image_folder_path  # JPG, PNG 등 확장자 맞춰서
    # 필요하면 jpg 등 다른 확장자도 처리 가능

    for fname in image_paths:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_shape is None:
            image_shape = gray.shape[::-1]  # (width, height)

        # 체커보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            # 코너를 더 정밀하게
            corners_sub = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            # 누적
            obj_points.append(objp)
            img_points.append(corners_sub)

    # 내부 파라미터, 왜곡 계수, 외부 파라미터 구하기
    if len(obj_points) < 1:
        print("체커보드 코너를 충분히 찾지 못하였습니다.")
        return None, None, None, None

    # flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3 등 필요에 따라 추가
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,  # 3D 실세계 점
        img_points,  # 2D 이미지 점
        image_shape,  # (width, height)
        None,  # 초기 camera_matrix
        None,  # 초기 dist_coeffs
    )

    if not ret:
        print("캘리브레이션이 제대로 수렴하지 않았습니다.")
        return None, None, None, None

    return camera_matrix, dist_coeffs, rvecs, tvecs



# Main Function
if __name__ == "__main__":
    # 캘리브레이션 데이터 로드
    data = json.load(open("data/calibrate_data.json"))
    robot_poses = np.array(data["poses"])

    robot_poses[:, :3] = robot_poses[:, :3]
    image_paths = ["data/" + d for d in data["file_name"]]

    checkerboard_size = (7, 10)  # 내부 코너 개수
    square_size = 25
    # 카메라 캘리브레이션 수행(내부 파라미터 왜곡 보정)
    camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera_from_chessboard(
        image_paths, checkerboard_size, square_size
    )

    R_gripper2base_list = []
    t_gripper2base_list = []
    R_camera2checker_list = []
    t_camera2checker_list = []
    R_checker2camera_list = []
    t_checker2camera_list = []

    for img_path, pose in zip(image_paths, robot_poses):
        # 1) 베이스->그리퍼 변환행렬
        T_base2gripper = get_robot_pose_matrix(*pose)

        # 2) 이미지 로딩
        image = cv2.imread(img_path)
        if image is None:
            continue

        # 3) 카메라->체커보드 변환 구하기
        R_cam2checker, t_cam2checker = find_checkerboard_pose(
            image, checkerboard_size, square_size, camera_matrix, dist_coeffs
        )
        if R_cam2checker is None:
            continue

        # T_gripper2base= np.linalg.inv(T_base2gripper)
        T_gripper2base= T_base2gripper

        R_gripper2base = T_gripper2base[:3, :3]
        t_gripper2base = T_gripper2base[:3, 3]

        R_gripper2base_list.append(R_gripper2base.copy())
        t_gripper2base_list.append(t_gripper2base.reshape(-1, 1).copy())

        T_cam2checker = np.eye(4)
        T_cam2checker[:3, :3] = R_cam2checker
        T_cam2checker[:3, 3] = t_cam2checker.flatten()
        
        T_checker2cam = T_cam2checker

        R_checker2camera_list.append(T_checker2cam[:3, :3].copy())
        t_checker2camera_list.append(T_checker2cam[:3, 3].copy())


    # Hand-Eye 캘리브레이션 수행
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base_list,
        t_gripper2base_list,
        R_checker2camera_list,
        t_checker2camera_list,
        method=cv2.CALIB_HAND_EYE_PARK,
    )


    T_base2gripper_example = get_robot_pose_matrix(*robot_poses[2])
    R_base2gripper_example = T_base2gripper_example[:3, :3]
    t_base2gripper_example = T_base2gripper_example[:3, 3]

    # 그리퍼->카메라 변환행렬
    T_gripper2cam = np.eye(4)
    T_gripper2cam[:3, :3] = R_cam2gripper
    T_gripper2cam[:3, 3] = t_cam2gripper.flatten()

    # 최종 베이스->카메라
    T_base2cam = T_base2gripper_example @ T_gripper2cam

    print("===== Hand-Eye Calibration Results =====")
    print("R_base2gripper:\n", T_base2gripper_example[:3, :3])
    print("T_base2gripper:\n", T_base2gripper_example[:3, 3])
    print("\n")
    print("R_base2camera:\n", T_base2cam[:3, :3])
    print("T_base2camera:\n", T_base2cam[:3, 3])
    print("\n")
    print("R_gripper2camera:\n", T_gripper2cam[:3, :3])
    print("T_gripper2camera:\n", T_gripper2cam[:3, 3].tolist())

    # save T_grigper2camera
    np.save("T_gripper2camera.npy", T_gripper2cam)
