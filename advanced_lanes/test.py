# Implementation of algorithm from https://stackoverflow.com/a/22640362/6029703
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from line_detection import transform_image

# https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/22640362#22640362


def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0] * len(y)
    stdFilter = [0] * len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i - 1]) > threshold * stdFilter[i - 1]:
            if y[i] > avgFilter[i - 1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i - 1]
            avgFilter[i] = np.mean(filteredY[(i - lag + 1) : i + 1])
            stdFilter[i] = np.std(filteredY[(i - lag + 1) : i + 1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i - lag + 1) : i + 1])
            stdFilter[i] = np.std(filteredY[(i - lag + 1) : i + 1])

    return dict(signals=np.asarray(signals), avgFilter=np.asarray(avgFilter), stdFilter=np.asarray(stdFilter))


test_image_path = Path("/home/bajic/Pictures/lila/1.jpg")
image = cv2.imread(str(test_image_path))
warp_matrix, img = transform_image(image)
y = histogram = np.sum(img[img.shape[0] // 2 :, :], axis=0)
# y = np.take(histogram, (1), axis=0)

# Settings: lag = 30, threshold = 5, influence = 0
lag = 30
threshold = 5
influence = 0

# Run algo with settings from above
result = thresholding_algo(histogram, lag=lag, threshold=threshold, influence=influence)

# Plot result
plt.subplot(211)
plt.plot(np.arange(1, len(y) + 1), y)

plt.plot(np.arange(1, len(y) + 1), result["avgFilter"], color="cyan", lw=2)

plt.plot(np.arange(1, len(y) + 1), result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)

plt.plot(np.arange(1, len(y) + 1), result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)

plt.subplot(212)
plt.step(np.arange(1, len(y) + 1), result["signals"], color="red", lw=2)
plt.ylim(-1.5, 1.5)
plt.show()