import cv2
import time
import numpy as np
from utils import load_calibration, project_cube


# -----------------------------
# Detect chessboard in frame
# -----------------------------
def detect_chessboard(image, board_w, board_h):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray_image, (board_w, board_h))

    if found:
        criteria = (cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, 30, 0.1)
        cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

    cv2.drawChessboardCorners(image, (board_w, board_h), corners, found)

    return found, image, corners


# -----------------------------
# Estimate pose of chessboard
# -----------------------------
def estimate_board_pose(corners, camera_matrix, distortion_coeffs, board_w, board_h):
    obj_points = np.zeros((board_w * board_h, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

    retval, rot_vec, trans_vec = cv2.solvePnP(
        obj_points, corners, camera_matrix, distortion_coeffs
    )

    return retval, rot_vec, trans_vec


# -----------------------------
# Main AR cube renderer
# -----------------------------
if __name__ == "__main__":
    BOARD_W, BOARD_H = 9, 6
    CUBE_SIZE = 2
    center_x = (BOARD_W - 1) / 2
    center_y = (BOARD_H - 1) / 2
    center = np.array([center_x, center_y, 0], dtype=np.float32)

    # 3D cube definition
    cube = np.float32([
        [0, 0, 0],
        [CUBE_SIZE, 0, 0],
        [CUBE_SIZE, CUBE_SIZE, 0],
        [0, CUBE_SIZE, 0],
        [0, 0, -CUBE_SIZE],
        [CUBE_SIZE, 0, -CUBE_SIZE],
        [CUBE_SIZE, CUBE_SIZE, -CUBE_SIZE],
        [0, CUBE_SIZE, -CUBE_SIZE],
    ]) + center

    _, cam_matrix, dist_coeffs = load_calibration("calibration.xml")

    cap = cv2.VideoCapture(0)
    angle = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        angle += 0.02

        # Rotation around Z-axis
        Rz = np.array([
            [np.cos(angle), np.sin(angle), 0],
            [-np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        rotated_cube = np.dot(cube - center, Rz.T) + center

        # Floating animation effect
        t = time.time()
        floating_offset = 0.5 * np.sin(t * 2)

        animated_cube = rotated_cube.copy()
        animated_cube[:, 2] += floating_offset

        found, frame, corners = detect_chessboard(frame, BOARD_W, BOARD_H)

        if found:
            ok_pose, rvec, tvec = estimate_board_pose(
                corners, cam_matrix, dist_coeffs, BOARD_W, BOARD_H
            )

            if ok_pose:
                frame = project_cube(
                    frame,
                    animated_cube,
                    rvec,
                    tvec,
                    cam_matrix,
                    dist_coeffs
                )

        cv2.imshow("Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()