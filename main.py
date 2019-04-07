import time
import io
import tempfile
from PIL import Image, ImageDraw
from cognitive_service import FaceAPI

import cv2
from cv2_camera_capture import CV2CameraCapture

from matplotlib import pyplot as plt


def draw_rect(dc, xy, color='red', width=5):
    (x1, y1), (x2, y2) = xy
    offset = 1
    for i in range(0, width):
        dc.rectangle(((x1, y1), (x2, y2)), outline=color)
        x1 -= offset
        y1 += offset
        x2 += offset
        y2 -= offset


class BufferWrapper:
    def __init__(self, buffer):
        self.buffer = buffer

    def read(self):
        return self.buffer


def main():
    with CV2CameraCapture(1).open() as camera:
        time.sleep(0.5)  # waiting for camera init.
        frame = camera.get_frame()
        img_gray = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        img_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        plt.imsave(buf, img_gray, format='png')
        image_data = BufferWrapper(buf.getvalue())
        emotions, rects = FaceAPI().detect(image_data).emotions()
        draw = ImageDraw.Draw(img_rgb)
        for rect in rects:
            draw_rect(draw, rect)
        for rect, emotion in zip(rects, emotions):
            draw.text((rect[0][0] + 20, rect[0][1] - 20), emotion, fill='green')

        plt.imshow(img_rgb)
        plt.show()


if __name__ == '__main__':
    main()
