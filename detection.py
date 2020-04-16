import cv2 as cv
import numpy as np

class ObjectDetector():
    def __init__(self):
        color = [0,127,255]
        self.kernel = np.asarray([[color]])
        self.threshold = 200

    def detect(self, img):
        map = 255 - np.mean(np.abs((img - self.kernel)), axis=2)
        _, map = cv.threshold(map.astype(np.uint8), 200, 255, cv.THRESH_BINARY)
        return cv.boundingRect(map)