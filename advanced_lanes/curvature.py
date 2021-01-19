import matplotlib.image as mpimg
import numpy as np

from line_detection import LaneDetector, transform_image
from utility import get_input_path


class ConversionRate:
    ym_per_pix = 25 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension


class RoadAndCar:
    def __init__(self, leftx, rightx, ploty):
        leftx = leftx[::-1]
        rightx = rightx[::-1]

        self.y_eval = np.max(ploty) * ConversionRate.ym_per_pix
        self.left_fit_cr = np.polyfit(ConversionRate.ym_per_pix * ploty, ConversionRate.xm_per_pix * leftx, 2)
        self.right_fit_cr = np.polyfit(ConversionRate.ym_per_pix * ploty, ConversionRate.xm_per_pix * rightx, 2)

    def measure_curvature_real(self):
        """
        Calculates the curvature of polynomial functions in meters.
        """
        left_curverad = np.power(
            (1 + (np.power(2 * self.left_fit_cr[0] * self.y_eval + self.left_fit_cr[1], 2))), 1.5
        ) / np.absolute(
            2 * self.left_fit_cr[0]
        )  ## Implement the calculation of the left line here
        right_curverad = np.power(
            (1 + (np.power(2 * self.right_fit_cr[0] * self.y_eval + self.right_fit_cr[1], 2))), 1.5
        ) / np.absolute(
            2 * self.right_fit_cr[0]
        )  ## Implement the calculation of the right line here

        return left_curverad, right_curverad

    def get_car_distance_text(self, img):
        middle_of_image = img.shape[1] / 2
        car_position = middle_of_image * ConversionRate.xm_per_pix

        height = img.shape[0] * ConversionRate.ym_per_pix
        left_line_base = self.left_fit_cr[0] * height ** 2 + self.left_fit_cr[1] * height + self.left_fit_cr[0]
        right_line_base = self.right_fit_cr[0] * height ** 2 + self.right_fit_cr[1] * height + self.right_fit_cr[0]
        lane_mid = (left_line_base + right_line_base) / 2

        dist_from_center = lane_mid - car_position
        if dist_from_center >= 0:
            center_text = "{} meters left of center".format(round(dist_from_center, 2))
        else:
            center_text = "{} meters right of center".format(round(-dist_from_center, 2))
        return center_text


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