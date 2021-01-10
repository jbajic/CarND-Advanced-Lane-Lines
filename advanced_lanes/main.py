import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from camera_distortion import load_camera_precalculated_coefficients, correct_camera_distortion
from threshold import get_binary_image
from perspective import warp_image, reverse_warp
from line_detection import sliding_window, get_polynominals
from utility import get_input_path, get_output_folder_path

# Import everything needed to edit/save/watch video clips
# from moviepy.editor import VideoFileClip
# from IPython.display import HTML


def weighted_img(img, initial_img, α=0.8, β=1.0, γ=0.0):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def draw_lines(img, warped_image, left_points, right_points)

    for line in left_line:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    for line in right_line:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def process_image():
    test_image_path = get_input_path("test_images").joinpath("test1.jpg")

    fig, ax = plt.subplots(4, 2)
    image = mpimg.imread(str(test_image_path))
    ax[0, 0].imshow(image)

    camera_matrix, distortion_coeff = load_camera_precalculated_coefficients()
    undistort_image = correct_camera_distortion(image, camera_matrix, distortion_coeff)
    ax[0, 1].imshow(undistort_image)

    binary_image = get_binary_image(undistort_image)
    ax[1, 0].imshow(binary_image)

    warped_image, warp_matrix = warp_image(binary_image)
    ax[1, 1].imshow(warped_image, cmap="gray")

    leftx, lefty, rightx, righty, lane_img = sliding_window(warped_image)
    left_fitx, right_fitx = get_polynominals(lane_img, leftx, lefty, rightx, righty)
    # Draw polynominals on new image
    tracked = np.zeroes_like(lane_img).astype(np.uint8)
    cv2.fillPoly(tracked, ,color=[255,0,0])

    ax[2, 0].imshow(lane_img)

    dewarped_image = reverse_warp(lane_img, warp_matrix)
    ax[2, 1].imshow(dewarped_image)

    merged_img = weighted_img(dewarped_image, image)
    ax[2, 2].imshow(merged_img)

    plt.show()


def main():
    process_image()
    # input_video = get_input_path("challenge_video.mp4")

    # clip1 = VideoFileClip(input_video)
    # white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!


if __name__ == "__main__":
    main()