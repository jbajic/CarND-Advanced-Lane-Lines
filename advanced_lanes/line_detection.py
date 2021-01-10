import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

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


def sliding_window(binary_warped, draw_windows=False):
    # Histogram of bottom half of image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2 :, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this

        # Draw the windows on the visualization image
        if draw_windows:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        ### Identify the nonzero pixels in x and y within the window ###
        good_left_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xleft_low)
            & (nonzerox < win_xleft_high)
        ).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xright_low)
            & (nonzerox < win_xright_high)
        ).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### f you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def get_polynominals(
    img,
    leftx,
    lefty,
    rightx,
    righty,
):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print("The function failed to fit a line!")
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    return left_fitx, right_fitx


def draw_polynominals(image, left_fitx, right_fitx, leftx, lefty, rightx, righty, save=False, img_name=""):
    ## Visualization ##
    # Colors in the left and right lane regions
    fig = plt.figure()
    fig.suptitle("Polynominals")
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    image[lefty, leftx] = [255, 0, 0]
    image[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.figure()
    plt.plot(left_fitx, ploty, color="yellow")
    plt.plot(right_fitx, ploty, color="yellow")
    if save:
        plt.savefig(str(get_output_folder_path().joinpath(f"{img_name}_line_detected.jpg")))
        plt.imshow(image)


def measure_curvature_real(img_shape, leftx, rightx):
    """
    Calculates the curvature of polynomial functions in meters.
    """
    leftx = leftx[::-1]
    rightx = rightx[::-1]
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
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

    for i, test_image_path in enumerate(test_images):
        test_image_path = get_input_path("test_images").joinpath("test1.jpg")
        image = mpimg.imread(str(test_image_path))
        # plt.imshow(image)
        warp_matrix, img = transform_image(image)

        # fig, ax = plt.subplots(2)
        plt.figure()
        leftx, lefty, rightx, righty, out_img = sliding_window(img)
        plt.imshow(out_img)

        left_fitx, right_fitx = get_polynominals(out_img, leftx, lefty, rightx, righty)
        polynominal_image = np.copy(out_img)

        left_curverad, right_curverad = measure_curvature_real(out_img.shape, left_fitx, right_fitx)
        print(f"Left line radius: {left_curverad} m, right lane radius: {right_curverad} m")

        if i == 0:
            draw_polynominals(
                polynominal_image,
                left_fitx,
                right_fitx,
                leftx,
                lefty,
                rightx,
                righty,
                True,
                img_name=test_image_path.stem,
            )
            # plt.imsave(str(get_output_folder_path().joinpath(f"{test_image_path.stem}.jpg")), image)
        plt.show()
        break


if __name__ == "__main__":
    main()
