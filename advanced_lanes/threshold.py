import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utility import get_input_dir_path, get_output_folder_path

def abs_sobel_thresh(img, orient="x", sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    if orient == "x":
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel_abs = np.absolute(sobel)

    sobel_scale = sobel_abs / np.max(sobel_abs)
    sobel_scaled = np.uint8(255 * sobel_scale)
    # Apply threshold
    grad_binary = np.zeros_like(sobel_scaled)
    grad_binary[(sobel_scaled > thresh[0]) & (sobel_scaled < thresh[1])] = 1

    return grad_binary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    scale_factor = sobel_magnitude / np.max(sobel_magnitude)
    sobel_scaled = np.uint8(255 * scale_factor)

    mag_binary = np.zeros_like(sobel_scaled)
    mag_binary[(sobel_scaled > mag_thresh[0]) & (sobel_scaled < mag_thresh[1])] = 1
    return mag_binary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Calculate gradient direction
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel_absx = np.absolute(sobelx)
    sobel_absy = np.absolute(sobely)
    sobel_direction = np.arctan2(sobel_absy, sobel_absx)

    # Apply threshold
    dir_binary = np.zeros_like(sobel_direction)
    dir_binary[(sobel_direction > thresh[0]) & (sobel_direction < thresh[1])] = 1
    return dir_binary

def get_binary_image(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    binary_image_saturation = hls[:, :, 2]
    # grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gradx = abs_sobel_thresh(binary_image_saturation, orient="x", sobel_kernel=9, thresh=(20, 100))
    grady = abs_sobel_thresh(binary_image_saturation, orient="y", sobel_kernel=9, thresh=(20, 100))
    mag_binary = mag_thresh(binary_image_saturation, sobel_kernel=9, mag_thresh=(30, 100))
    dir_binary = dir_threshold(binary_image_saturation, sobel_kernel=15, thresh=(0.7, 1.3))
    combined_binary = np.zeros_like(dir_binary)
    # combined_binary[((gradx == 1) & (dir_binary == 1)) | ((mag_binary == 1) & (grady == 1))] = 1
    combined_binary[((gradx == 1) & (dir_binary == 1) | ((mag_binary == 1) & (grady == 1)))] = 1

    return combined_binary

def show_tresholding_stages(loaded_img, gradx, grady, mag_binary, dir_binary, combined_binary):
    figure, axes = plt.subplots(3, 2)

    axes[0, 0].imshow(loaded_img)
    axes[0, 0].set_title("Normal image")

    axes[0, 1].imshow(gradx, cmap="gray")
    axes[0, 1].set_title("Absolute gradient x")

    axes[1, 0].imshow(grady, cmap="gray")
    axes[1, 0].set_title("Absolute gradient y")

    axes[1, 1].imshow(mag_binary, cmap="gray")
    axes[1, 1].set_title("Magnitude gradient")

    axes[2, 0].imshow(dir_binary, cmap="gray")
    axes[2, 0].set_title("Direction gradient")

    axes[2, 1].imshow(combined_binary, cmap="gray")
    axes[2, 1].set_title("Combined gradients")
    plt.show()

def main():
    # test_image_path = get_input_dir_path("test_images").joinpath("test1.jpg")
    test_images = get_input_dir_path("test_images").glob("*.jpg")
    for i, test_image_path in enumerate(test_images):
        loaded_img = mpimg.imread(str(test_image_path))
        binary_image = get_binary_image(loaded_img)

        # show_tresholding_stages(loaded_img, gradx, grady, mag_binary, dir_binary, combined_binary)
        fig, ax = plt.subplots(2)
        ax[0].imshow(loaded_img)
        ax[0].set_title("Loaded Image")
        ax[1].imshow(binary_image, cmap="gray")
        ax[1].set_title("Binary Image")
        if i == 0:
            plt.imsave(str(get_output_folder_path().joinpath(f"{test_image_path.stem}.jpg")), loaded_img)
            plt.imsave(str(get_output_folder_path().joinpath(f"{test_image_path.stem}_binary.jpg")), binary_image, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()