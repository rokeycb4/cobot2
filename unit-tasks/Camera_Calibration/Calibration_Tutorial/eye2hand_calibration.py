import json
from scipy.spatial.transform import Rotation
import numpy as np
import cv2

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

# 체커보드 이미지를 이용한 카메라 보정
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

from scipy.linalg import sqrtm
from numpy.linalg import inv

# 4) 여러 개의 변환 행렬 조합 함수
def compose_transformation_matrices(R_list, t_list):
    T_list = []
    for R, t in zip(R_list, t_list):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.ravel(t)  # t가 벡터 형태여야 합니다.
        T_list.append(T)
    return T_list

# 회전행렬을 로그변환하는 함수
def logR(T):
    R = T[0:3, 0:3]
    theta = np.arccos((np.trace(R) - 1) / 2)
    logr = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) * theta / (2 * np.sin(theta))
    return logr

# A와 B의 변환을 이용하여 보정 행렬 계산
def Calibrate(A, B):
    n_data = len(A)
    M = np.zeros((3, 3))

    for i in range(n_data - 1):
        alpha  = logR(A[i])
        beta   = logR(B[i])
        alpha2 = logR(A[i + 1])
        beta2  = logR(B[i + 1])
        
        alpha3 = np.cross(alpha, alpha2)
        beta3  = np.cross(beta, beta2)
        
        M1 = np.dot(beta.reshape(3, 1), alpha.reshape(1, 3))
        M2 = np.dot(beta2.reshape(3, 1), alpha2.reshape(1, 3))
        M3 = np.dot(beta3.reshape(3, 1), alpha3.reshape(1, 3))
        
        M += M1 + M2 + M3

    theta = np.dot(sqrtm(inv(np.dot(M.T, M))), M.T)

    C = np.zeros((3 * n_data, 3))
    d = np.zeros((3 * n_data, 1))
    for i in range(n_data):
        rot_a = A[i][:3, :3]
        trans_a = A[i][:3, 3]
        trans_b = B[i][:3, 3]
        C[3 * i:3 * i + 3, :] = np.eye(3) - rot_a
        d[3 * i:3 * i + 3, 0] = trans_a - np.dot(theta, trans_b)
    
    b_x = np.dot(inv(np.dot(C.T, C)), np.dot(C.T, d))
    return theta, b_x

# Main Function
if __name__ == "__main__":
    data = json.load(open("data/calibrate_data.json"))
    robot_poses = np.array(data["poses"])

    robot_poses[:, :3] = robot_poses[:, :3]
    image_paths = ["data/" + d for d in data["file_name"]]

    valid_indices = []
    for i, pose in enumerate(robot_poses):
        T_base2gripper = get_robot_pose_matrix(*pose)
        det_T = np.linalg.det(T_base2gripper)
        print(f"Index {i}: det(T_base2gripper) = {det_T}")

        if np.abs(det_T) > 1e-6:
            valid_indices.append(i)
        else:
            print(f"⚠️ Warning: Singular T_base2gripper at index {i}!")

    robot_poses = robot_poses[valid_indices]
    image_paths = [image_paths[i] for i in valid_indices]

    checkerboard_size = (8, 6)  # 내부 코너 개수
    square_size = 25

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

        T_gripper2base= np.linalg.inv(T_base2gripper)

        R_gripper2base = T_gripper2base[:3, :3]
        t_gripper2base = T_gripper2base[:3, 3]

        R_gripper2base_list.append(R_gripper2base.copy())
        t_gripper2base_list.append(t_gripper2base.reshape(-1, 1).copy())

        T_cam2checker = np.eye(4)
        T_cam2checker[:3, :3] = R_cam2checker
        T_cam2checker[:3, 3] = t_cam2checker.flatten()
        T_checker2cam = np.linalg.inv(T_cam2checker)

        R_checker2camera_list.append(T_checker2cam[:3, :3].copy())
        t_checker2camera_list.append(T_checker2cam[:3, 3].copy())

    T_gripper2base_list = compose_transformation_matrices(R_gripper2base_list, t_gripper2base_list)
    T_checker2cam_list = compose_transformation_matrices(R_checker2camera_list, t_checker2camera_list)
    A_list = []
    B_list = []
    num_pairs = min(len(T_gripper2base_list), len(T_checker2cam_list))

    for i, T in enumerate(T_gripper2base_list):
        det = np.linalg.det(T)
        if np.abs(det) < 1e-6:
            print(f"⚠️ Warning: T_gripper2base_list[{i}] is singular or nearly singular!")

    for i in range(num_pairs - 1):
        A_i = np.dot(inv(T_gripper2base_list[i]), T_gripper2base_list[i + 1])
        B_i = np.dot(inv(T_checker2cam_list[i]), T_checker2cam_list[i + 1])
        A_list.append(A_i)
        B_list.append(B_i)

    theta, b_x = Calibrate(A_list, B_list)
    X = np.eye(4)
    X[:3, :3] = theta
    X[:3, 3] = b_x.flatten()
    T_cam2base = X
    print(T_cam2base)
    print(T_cam2base[:3, 3])
    np.save("T_cam2base.npy", T_cam2base)
