import matplotlib.image as mpimg
import numpy as np

from line_detection import LaneDetector, transform_image
from utility import get_input_path


def measure_curvature_real(leftx, rightx, ploty):
    """
    Calculates the curvature of polynomial functions in meters.
    """
    leftx = leftx[::-1]
    rightx = rightx[::-1]
    # # Define conversions in x and y from pixels space to meters
    # ym_per_pix = 30 / 720  # meters per pixel in y dimension
    # xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    ym_per_pix = 25 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 800  # meters per pixel in x dimension

    y_eval = np.max(ploty) * ym_per_pix
    left_fit_cr = np.polyfit(ym_per_pix * ploty, xm_per_pix * leftx, 2)
    right_fit_cr = np.polyfit(ym_per_pix * ploty, xm_per_pix * rightx, 2)

    left_curverad = np.power((1 + (np.power(2 * left_fit_cr[0] * y_eval + left_fit_cr[1], 2))), 1.5) / np.absolute(
        2 * left_fit_cr[0]
    )  ## Implement the calculation of the left line here
    right_curverad = np.power((1 + (np.power(2 * right_fit_cr[0] * y_eval + right_fit_cr[1], 2))), 1.5) / np.absolute(
        2 * right_fit_cr[0]
    )  ## Implement the calculation of the right line here

    return left_curverad, right_curverad


def main():
    test_images = get_input_path("test_images").glob("*.jpg")
    lane_detector = LaneDetector()

    for i, test_image_path in enumerate(test_images):
        test_image_path = get_input_path("test_images").joinpath("test1.jpg")
        image = mpimg.imread(str(test_image_path))
        warp_matrix, img = transform_image(image)

        leftx, lefty, rightx, righty, out_img = lane_detector.sliding_window(img)

        left_fitx, right_fitx, plotpy = lane_detector.get_polynominals(out_img, leftx, lefty, rightx, righty)
        polynominal_image = np.copy(out_img)

        left_curverad, right_curverad = measure_curvature_real(left_fitx, right_fitx, plotpy)
        print(f"Left line radius: {left_curverad} m, right lane radius: {right_curverad} m")

        break


if __name__ == "__main__":
    main()