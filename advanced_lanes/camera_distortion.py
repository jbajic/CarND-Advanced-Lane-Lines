import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from collections import namedtuple
from pathlib import Path
from typing import Tuple

from utility import get_output_folder_path


CAMERA_COEFFICIENTS_FILE = "camera_coefficients.pickle"
CameraCoefficients = namedtuple("CameraCoefficients", ["matrix", "distortion"])


def save_camera_coefficients(mtx, dist):
    camera_coefficients = CameraCoefficients(mtx, dist)

    with open(CAMERA_COEFFICIENTS_FILE, "wb") as file:
        pickle.dump(camera_coefficients, file)


def load_camera_precalculated_coefficients():
    camera_coefficients_path = Path(CAMERA_COEFFICIENTS_FILE)

    if camera_coefficients_path.exists():
        camera_coefficients = {}
        with camera_coefficients_path.open("rb") as file:
            camera_coefficients = pickle.load(file)
        return camera_coefficients.matrix, camera_coefficients.distortion
    print("Create camera coefficients!")


def get_camera_calibration_images() -> Path:
    return Path(__file__).resolve().parents[1].joinpath("camera_cal").glob("*.jpg")


def calculate_camera_coefficients(show_results=False):
    chessboard_images = get_camera_calibration_images()
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    nx = 9
    ny = 6
    img_points = []  # 2d points in image space
    obj_points = []  # 3d points in real world space
    (width, height) = None, None

    # prepare object points
    for image in chessboard_images:
        img = cv2.imread(str(image))
        if width is None:
            height, width = img.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret == True:
            obj_points.append(objp)
            img_points.append(corners)

        if show_results:
            img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            cv2.imshow(image.name, img)
            cv2.waitKey(500)
    if show_results:
        cv2.destroyAllWindows()

    # cv2.calibrateCamera returns (https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d):
    #  - ret => RMS (root mean squared) re-projection error
    #  - mtx => camera matrix
    #  - dist => distortion coefficients
    #  - rvecs => Output vector of rotation vectors
    #  - tvecs => Output vector of translation vectors
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (width, height), None, None)
    return ret, mtx, dist


def correct_camera_distortion(img, matrix, dist_coefficients):
    return cv2.undistort(img, matrix, dist_coefficients, None, matrix)


def main():
    """
    Main function it an example function
    """
    ret, mtx, dist = calculate_camera_coefficients()
    print(f"RMS: {ret}")
    print(f"Camera matrix: {mtx}")
    print(f"Distortion coefficients: {dist}")

    save_camera_coefficients(mtx, dist)

    img = mpimg.imread("/home/bajic/Projects/Udacity/projects/CarND-Advanced-Lane-Lines/camera_cal/calibration1.jpg")
    undistorted = correct_camera_distortion(img, mtx, dist)

    output_path_undistorted = get_output_folder_path().joinpath("calibration1_undistorted.jpg")
    output_path_distorted = get_output_folder_path().joinpath("calibration1.jpg")

    cv2.imwrite(str(output_path_distorted), img)
    cv2.imwrite(str(output_path_undistorted), undistorted)
    cv2.imshow("calibration1", undistorted)
    cv2.waitKey(1000)


if __name__ == "__main__":
    main()
