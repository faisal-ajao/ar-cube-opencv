import cv2
import numpy as np


# -----------------------------
# Save calibration data to file
# -----------------------------
def save_calibration(filename, image_size, intrinsic_matrix, distortion_coeffs):
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)

    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("camera_matrix", intrinsic_matrix)
    fs.write("distortion_coefficients", distortion_coeffs)

    fs.release()
    print(f"Calibration saved to {filename}")


# -----------------------------
# Load calibration data from file
# -----------------------------
def load_calibration(filename):
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

    image_width = int(fs.getNode("image_width").real())
    image_height = int(fs.getNode("image_height").real())

    camera_matrix = fs.getNode("camera_matrix").mat()
    distortion_coeffs = fs.getNode("distortion_coefficients").mat()

    fs.release()

    return (image_width, image_height), camera_matrix, distortion_coeffs


# -----------------------------
# Project and draw 3D cube onto image
# -----------------------------
def project_cube(image, cube, rot_vec, trans_vec, camera_matrix, distortion_coeffs):
    img_points, _ = cv2.projectPoints(
        cube, rot_vec, trans_vec, camera_matrix, distortion_coeffs
    )

    img_points = img_points.reshape(-1, 2).astype(int)

    # Bottom face (cool teal)
    cv2.drawContours(image, [img_points[:4]], -1, (0, 200, 200), -1)

    # Vertical edges
    for i, j in zip(range(4), range(4, 8)):
        cv2.line(image, tuple(img_points[i]), tuple(img_points[j]), (255, 255, 255), 2)

    # Top face (purple tone)
    cv2.drawContours(image, [img_points[4:]], -1, (200, 0, 200), -1)

    # Outline edges (soft gray)
    cv2.polylines(image, [img_points[:4]], True, (200, 200, 200), 2)
    cv2.polylines(image, [img_points[4:]], True, (200, 200, 200), 2)

    return image