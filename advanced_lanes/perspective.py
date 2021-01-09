import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utility import get_input_dir_path, get_output_folder_path


def warp_image(img):
    img_size = (img.shape[1], img.shape[0])
    # src_points = np.float32(
    #     [
    #         [(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    #         [((img_size[0] / 6) - 10), img_size[1] - 50],
    #         [(img_size[0] * 5 / 6) + 60, img_size[1] - 50],
    #         [(img_size[0] / 2 + 55), img_size[1] / 2 + 100],
    #     ]
    # )
    # dest_points = np.float32(
    #     [
    #         [(img_size[0] / 4), 0],
    #         [(img_size[0] / 4), img_size[1]],
    #         [(img_size[0] * 3 / 4), img_size[1]],
    #         [(img_size[0] * 3 / 4), 0],
    #     ]
    # )
    src_points = np.array(
        [
            [607, 450],  # top left
            [315, 670],  # bottom left
            [1080, 670],  # bottom right
            [725, 450],  # top right
        ],
        dtype=np.float32,
    )
    offset = 800
    dest_points = np.float32(
        [
            [(img_size[0] / 4), 0],
            [(img_size[0] / 4), img_size[1]],
            [(img_size[0] * 3 / 4), img_size[1]],
            [(img_size[0] * 3 / 4), 0],
        ]
    )
    # for p in src_points:
    #     cv2.circle(img, tuple(p), radius=0, color=(0, 0, 255), thickness=10)

    M = cv2.getPerspectiveTransform(src_points, dest_points)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M


if __name__ == "__main__":
    # test_image_path = get_input_dir_path("test_images").joinpath("test5.jpg")
    test_images = get_input_dir_path("test_images").glob("*.jpg")

    for i, test_image_path in enumerate(test_images):
        img = cv2.imread(str(test_image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        warped, M = warp_image(img_rgb)
        fig, ax = plt.subplots(2)
        ax[0].imshow(img_rgb)
        ax[0].set_title("Unwarped image")
        ax[1].imshow(warped)
        ax[1].set_title("Warped image")

        if i == 0:
            plt.imsave(str(get_output_folder_path().joinpath(f"{test_image_path.stem}.jpg")), img)
            plt.imsave(str(get_output_folder_path().joinpath(f"{test_image_path.stem}_warped.jpg")), warped)
    plt.show()
