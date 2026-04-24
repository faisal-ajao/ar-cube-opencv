import os
import time
import cv2
import numpy as np
from utils import load_calibration, save_calibration


# -----------------------------
# Capture chessboard images for calibration
# -----------------------------
def capture_chessboard_images(board_w, board_h, n_boards, delay=1.0, image_scale=0.5):
    board_size = (board_w, board_h)

    object_points = []
    image_points = []

    # 3D points in real world space (Z=0 plane)
    objp = np.zeros((board_w * board_h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    last_capture = 0
    image_size = None

    print(f"Starting capture for {n_boards} chessboard images")

    while len(image_points) < n_boards:
        ret, frame = cap.read()
        if not ret:
            break

        # Store image size once
        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])

        # Resize for faster detection
        resized = cv2.resize(frame, None, fx=image_scale, fy=image_scale)
        resized_copy = np.copy(resized)

        # Detect chessboard corners
        found, corners = cv2.findChessboardCorners(resized, board_size)

        # Draw detected corners
        cv2.drawChessboardCorners(resized, board_size, corners, found)

        # Save valid detections with delay spacing
        if found and time.time() - last_capture > delay:
            cv2.imwrite(
                f"chessboard_images/chessboard_{len(image_points)}.png",
                resized_copy
            )
            last_capture = time.time()

            corners = corners / image_scale
            image_points.append(corners)
            object_points.append(objp.copy())

            print(f"Captured {len(image_points)}/{n_boards} images.")

        cv2.imshow("Calibration", resized)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    return object_points, image_points, image_size


# -----------------------------
# Camera calibration
# -----------------------------
def calibrate_camera(object_points, image_points, image_size):
    ret, intrinsic_matrix, distortion_coeffs, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_PRINCIPAL_POINT,
    )

    print(f"Calibration done. Reprojection error: {ret:.4f}")
    return intrinsic_matrix, distortion_coeffs


# -----------------------------
# Live undistortion preview
# -----------------------------
def undistort_live(camera_matrix, distortion_coeffs, image_size):
    cap = cv2.VideoCapture(0)

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion_coeffs,
        None,
        camera_matrix,
        image_size,
        cv2.CV_16SC2
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        cv2.imshow("Undistorted", undistorted)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    BOARD_W, BOARD_H = 9, 6
    N_BOARDS = 15
    DELAY = 1.0
    IMAGE_SCALE = 0.5

    IMAGES_FOLDER = "chessboard_images"
    os.makedirs(IMAGES_FOLDER, exist_ok=True)

    obj_pts, img_pts, img_size = capture_chessboard_images(
        BOARD_W, BOARD_H, N_BOARDS, DELAY, IMAGE_SCALE
    )

    cam_matrix, dist_coeffs = calibrate_camera(obj_pts, img_pts, img_size)

    save_calibration("calibration.xml", img_size, cam_matrix, dist_coeffs)

    _, loaded_cam_matrix, loaded_dist_coeffs = load_calibration("calibration.xml")

    undistort_live(loaded_cam_matrix, loaded_dist_coeffs, img_size)