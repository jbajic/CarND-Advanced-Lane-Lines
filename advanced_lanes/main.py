import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from camera_distortion import load_camera_precalculated_coefficients, correct_camera_distortion
from curvature import ConversionRate, RoadAndCar
from threshold import get_binary_image
from perspective import warp_image, reverse_warp
from line_detection import LaneDetector
from utility import get_input_path, get_output_folder_path

curvature_radiuses = []

def weighted_img(img, initial_img, α=1.0, β=1.0, γ=0.0):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def draw_lane(warped_img, left_fitx, right_fitx, ploty, warp_matrix):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img, dtype=np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))], dtype=np.int32)
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # cv2.polylines(color_warp, pts_left, True, (255, 0, 0), thickness=30)
    # cv2.polylines(color_warp, pts_right, True, (255, 0, 0), thickness=30)
    # Reverse warp
    newwarp = reverse_warp(color_warp, warp_matrix)

    return newwarp

curvature_radiuses = []

def process_image(image, lane_detector, camera_matrix, distortion_coeff):
    undistort_image = correct_camera_distortion(image, camera_matrix, distortion_coeff)

    binary_image = get_binary_image(undistort_image)

    warped_image, warp_matrix = warp_image(binary_image)

    left_fitx, right_fitx, ploty, out_img = lane_detector.get_lanes(warped_image)
    dewarped_lane_image = draw_lane(warped_image, left_fitx, right_fitx, ploty, warp_matrix)

    # dewarped_lane_image = reverse_warp(out_img, warp_matrix)
    merged_img = weighted_img(dewarped_lane_image, image, β=0.8)

    road_and_car_parameters = RoadAndCar(left_fitx, right_fitx, ploty)
    left_curverad, right_curverad = road_and_car_parameters.measure_curvature_real()

    global curvature_radiuses
    curvature_radius = round(np.mean([left_curverad, right_curverad]), 0)
    if len(curvature_radiuses) > 10:
        curvature_radius_mean = np.mean(curvature_radiuses)
        if curvature_radius < curvature_radius_mean * 3:
            curvature_radiuses.append(curvature_radius)
            curvature_radiuses.pop(0)
        else:
            curvature_radius = round(curvature_radius_mean, 0)
        curvature_radius = round(np.mean(curvature_radiuses), 0)
    else:
        curvature_radiuses.append(curvature_radius)

    curvature_text = f"Radius of Curvature = {curvature_radius} m"
    cat_distance_text = road_and_car_parameters.get_car_distance_text(merged_img)

    cv2.putText(merged_img, curvature_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(merged_img, cat_distance_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return merged_img


def test_main():
    # test_image_path = get_input_path("test_images").joinpath("test1.jpg")
    # test_images = get_input_path("test_images").glob("*.jpg")
    from pathlib import Path

    test_images = Path("/home/bajic/Pictures/lila/").glob("*.jpg")
    lane_detector = LaneDetector()

    for i, test_image_path in enumerate(test_images):
        fig, ax = plt.subplots(3, 2)
        image = mpimg.imread(str(test_image_path))
        ax[0, 0].imshow(image)

        camera_matrix, distortion_coeff = load_camera_precalculated_coefficients()
        undistort_image = correct_camera_distortion(image, camera_matrix, distortion_coeff)
        ax[0, 1].imshow(undistort_image)

        binary_image = get_binary_image(undistort_image)
        ax[1, 0].imshow(binary_image, cmap="gray")

        warped_image, warp_matrix = warp_image(binary_image)
        ax[1, 1].imshow(warped_image, cmap="gray")

        leftx, lefty, rightx, righty, lane_img = lane_detector.sliding_window(warped_image)
        left_fitx, right_fitx, ploty = lane_detector.get_polynominals(lane_img, leftx, lefty, rightx, righty)
        ax[2, 0].imshow(lane_img)

        dewarped_lane_image = draw_lane(warped_image, left_fitx, right_fitx, ploty, warp_matrix)
        merged_img = weighted_img(dewarped_lane_image, image, β=0.8)
        ax[2, 1].imshow(merged_img)

        plt.show()
        break


def main():
    # test_main()
    input_video = str(get_input_path("project_video.mp4"))
    tracked_video = str(get_output_folder_path().joinpath("tracked_video.mp4"))

    camera_matrix, distortion_coeff = load_camera_precalculated_coefficients()
    lane_detector = LaneDetector()
    clip1 = VideoFileClip(input_video)
    white_clip = clip1.fl_image(
        lambda image: process_image(image, lane_detector, camera_matrix, distortion_coeff)
    )  # NOTE: this function expects color images!!

    white_clip.write_videofile(tracked_video, audio=False)


if __name__ == "__main__":
    main()