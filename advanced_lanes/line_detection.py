import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from dataclasses import dataclass, field
from typing import List

from camera_distortion import load_camera_precalculated_coefficients, correct_camera_distortion
from threshold import get_binary_image
from perspective import warp_image
from utility import get_input_path, get_output_folder_path


def transform_image(image, show_process=False):
    camera_matrix, distortion_coeff = load_camera_precalculated_coefficients()
    undistort_image = correct_camera_distortion(image, camera_matrix, distortion_coeff)

    binary_image = get_binary_image(undistort_image)
    binary_warped, M = warp_image(binary_image)

    if show_process:
        fig, ax = plt.subplots(3)
        ax[0].imshow(undistort_image)
        ax[1].imshow(binary_image, cmap="gray")
        ax[2].imshow(binary_warped, cmap="gray")
        plt.show()

    return M, binary_warped


@dataclass
class LanePolynominal:
    current_fit: List[float] = None
    previous_fits: List[List[float]] = field(default_factory=list)
    number_of_wrong_iterations: int = 0
    detected: bool = False

    def fit_polynominal(self, lanex, laney, side):
        try:
            self.current_fit = np.polyfit(laney, lanex, 2)
            self.previous_fits.append(self.current_fit)

            best_fit = np.mean(self.previous_fits[-5:], axis=0)
            self.current_fit = best_fit

        except Exception:
            if len(self.previous_fits) == 1 or self.number_of_wrong_iterations >= 5:
                self.previous_fits = []
                number_of_wrong_iterations = 0

class LaneDetector:
    LEFT_LANE = "left"
    RIGHT_LANE = "right"

    def __init__(self):
        self.left_lane = LanePolynominal()
        self.right_lane = LanePolynominal()
        # Set Hyperparamaters
        self.nwindows = 20
        # Set the width of the windows +/- margin
        self.margin = 75
        # Set minimum number of pixels found to recenter window
        self.minpix = 50
        # Set height of windows - based on nwindows above and image shape

    def _get_lane_centers(self, binary_warped, draw_windows=False):
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2 :, :], axis=0)
        if draw_windows:
            plt.figure()
            plt.plot(histogram)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        return leftx_base, rightx_base

    def get_lanes(self, binary_warped, draw=False):
        # left_fit = self.left_fits[-1] if self.left_fits is not [] and self.left_fits[-1] != None else None
        # right_fit = self.right_fits[-1] if self.right_fits is not [] and self.right_fits[-1] != None else None
        if self.left_lane.detected is True and self.right_lane.detected is True:
            left_fitx, right_fitx, ploty = self._search_around_prior_poly(binary_warped, draw)

        self.window_height = np.int(binary_warped.shape[0] // self.nwindows)
        self.left_lane_tracker = True
        self.right_lane_tracker = True
        windows_img = np.dstack((binary_warped, binary_warped, binary_warped))
        width, height = height, width = binary_warped.shape[:2]

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_base, rightx_base = self._get_lane_centers(binary_warped)

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        leftx_current = leftx_base
        rightx_current = rightx_base
        counter = 0

        for window in range(self.nwindows):
            if self.left_lane_tracker:
                good_left_inds, leftx_current = self._sliding_window(
                    window,
                    leftx_current,
                    nonzerox,
                    nonzeroy,
                    height,
                    width,
                    "left",
                    windows_img,
                    counter,
                    draw_windows=draw,
                )
                left_lane_inds.append(good_left_inds)
            if self.right_lane_tracker:
                good_right_inds, rightx_current = self._sliding_window(
                    window,
                    rightx_current,
                    nonzerox,
                    nonzeroy,
                    height,
                    width,
                    "right",
                    windows_img,
                    counter,
                    draw_windows=draw,
                )
                right_lane_inds.append(good_right_inds)
            if self.left_lane_tracker and self.right_lane_tracker:
                counter += 1
            if not self.left_lane_tracker and not self.right_lane_tracker:
                break

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if draw:
            fig = plt.figure()
            fig.suptitle("Windows")
            plt.imshow(windows_img)

        left_fitx, right_fitx, ploty = self._get_polynominals(binary_warped, leftx, lefty, rightx, righty)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        return left_fitx, right_fitx, ploty, out_img

    def _sliding_window(
        self, window, lanex_current, nonzerox, nonzeroy, height, width, side, out_img, counter, draw_windows=False
    ):
        win_y_low = height - (window + 1) * self.window_height
        win_y_high = height - window * self.window_height
        win_xlane_low = lanex_current - self.margin
        win_xlane_high = lanex_current + self.margin

        if draw_windows:
            cv2.rectangle(out_img, (win_xlane_low, win_y_low), (win_xlane_high, win_y_high), (0, 255, 0), 2)

        good_lane_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xlane_low)
            & (nonzerox < win_xlane_high)
        ).nonzero()[0]

        if (win_xlane_high > width or win_xlane_low < 0) and counter >= 3:
            if side == self.LEFT_LANE:
                self.left_lane_tracker = False
            elif side == self.RIGHT_LANE:
                self.right_lane_tracker = False

        ### f you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_lane_inds) > self.minpix:
            lanex_current = np.int(np.mean(nonzerox[good_lane_inds]))

        return good_lane_inds, lanex_current

    def _search_around_prior_poly(self, binary_warped, draw_windows=False):
        # Take last polynominals
        left_fit = self.left_fits[-1]
        right_fit = self.right_fits[-1]

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = (
            nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - self.margin)
        ) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + self.margin))
        right_lane_inds = (
            nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - self.margin)
        ) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + self.margin))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fitx, right_fitx, ploty = self._get_polynominals(binary_warped, leftx, lefty, rightx, righty)

        return left_fitx, right_fitx, ploty

    def _get_polynominals(
        self,
        img,
        leftx,
        lefty,
        rightx,
        righty,
    ):
        self.left_lane.fit_polynominal(leftx, lefty, self.LEFT_LANE)
        self.right_lane.fit_polynominal(rightx, righty, self.RIGHT_LANE)
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

        try:
            left_fitx = (
                self.left_lane.current_fit[0] * ploty ** 2
                + self.left_lane.current_fit[1] * ploty
                + self.left_lane.current_fit[2]
            )
            right_fitx = (
                self.right_lane.current_fit[0] * ploty ** 2
                + self.right_lane.current_fit[1] * ploty
                + self.right_lane.current_fit[2]
            )
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print("The function failed to fit a line!")
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

        return left_fitx, right_fitx, ploty


def main():
    test_images = get_input_path("test_images").glob("*.jpg")
    from pathlib import Path

    # test_images = Path("/home/bajic/Pictures/lila/").glob("*.jpg")
    lane_detector = LaneDetector()

    for i, test_image_path in enumerate(test_images):
        test_image_path = Path("/home/bajic/Pictures/lila/1.jpg")
        # test_image_path = get_input_path("test_images").joinpath("test1.jpg")
        image = mpimg.imread(str(test_image_path))
        plt.figure
        plt.imshow(image)
        warp_matrix, img = transform_image(image)

        left_fitx, right_fitx, ploty, out_img = lane_detector.get_lanes(img, draw=True)

        # left_fitx, right_fitx, ploty = lane_detector.get_polynominals(out_img, leftx, lefty, rightx, righty)
        fig = plt.figure()
        fig.suptitle("Polylines")
        plt.imshow(out_img)

        # plt.imsave(str(get_output_folder_path().joinpath(f"{test_image_path.stem}.jpg")), image)
        plt.show()
        break


if __name__ == "__main__":
    main()
