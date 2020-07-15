import cv2 as cv
import numpy as np

class ObjectDetector():
    """Класс, отвечающий за обнаружение объекта.

    Attributes
    -------
    kernel : ndarray
        RGB-цвет детектируемого объекта в виде массива с формой (1,3)
    threshold : int
        Порог принадлежности точки объекту

    Methods
    -------
    detect(img)
        Определяет обрамляющий прямоугольник объекта с данным цветом
    """

    def __init__(self):
        color = [0,127,255] # голубой
        self.kernel = np.asarray([[color]])
        self.threshold = 200

    def detect(self, img):
        """Определяет обрамляющий прямоугольник объекта с данным цветом

        Детектирование на основе цветового сопоставления по L1-норме

        Parameters
        ----------
        img : ndarray
            Входное трёхканальное изображение с цветовой моделью RGB

        Returns
        -------
        int
            Координата по горизонтали
        int
            Координата по вертикали
        int
            Ширина
        int
            Высота
        """

        map = 255 - np.mean(np.abs((img - self.kernel)), axis=2)
        _, map = cv.threshold(map.astype(np.uint8), 200, 255, cv.THRESH_BINARY)
        return cv.boundingRect(map)