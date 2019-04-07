import cv2
import time
from PIL import Image
from matplotlib import pyplot as plt
from contextlib import contextmanager


class CV2CameraCapture:

    def __init__(self, device=0):
        self.camera = cv2.VideoCapture(device)
        if not self.camera.isOpened():
            self.camera.open()

    @contextmanager
    def open(self):
        try:
            yield self
        finally:
            self.camera.release()

    def get_frame(self):
        _, frame = self.camera.read()
        return frame


if __name__ == "__main__":
    with CV2CameraCapture(0).open() as camera:
        time.sleep(3)
        img = Image.fromarray(camera.get_frame())
        plt.imshow(img)
        plt.show()
