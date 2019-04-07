import tempfile
from PIL import Image
from cognitive_service import FaceAPI

import cv2
from cv2_camera_capture import CV2CameraCapture

from matplotlib import pyplot as plt


def main():
    with CV2CameraCapture(0).open() as camera:
        cv2.imshow('frame', camera.get_frame())
        cv2.waitKey(0)
        # img = Image.fromarray(camera.get_frame())
        # temp_file = tempfile.NamedTemporaryFile(suffix='.png')
        # print(temp_file.name)
        # with open(temp_file.name, 'wb') as f:
        #     img.save(f)
        #     img_reread = Image.open(temp_file)
        #     plt.imshow(img_reread)
        #     plt.show()
        # print(FaceAPI().detect(temp_file.name).emotions())
        # # temp_file.close()
        # # temp_file.close()


if __name__ == '__main__':
    main()
