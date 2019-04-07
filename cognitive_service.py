import json
import cognitive_face as CF


class FaceAPI:

    def __init__(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        self.CF = CF
        self.CF.Key.set(self.config['api'])
        self.CF.BaseUrl.set(self.config['endpoint'])
        self.faces = None

    def detect(self, image):
        self.faces = self.CF.face.detect(image, attributes='emotion')
        return self

    def emotions(self):
        def _get_top_emotion():
            attributes = face['faceAttributes']['emotion']
            top_emotion = sorted(attributes, key=attributes.get)[-1]
            return top_emotion

        emotions = []
        for face in self.faces:
            emotions.append(_get_top_emotion())
        return emotions, self._rects()

    def _rects(self):
        def _parse_rectangle():
            rect = face['faceRectangle']
            left = rect['left']
            top = rect['top']
            bottom = left + rect['height']
            right = top + rect['width']
            return (left, top), (bottom, right)

        rects = []
        for face in self.faces:
            rects.append(_parse_rectangle())
        return rects


if __name__ == "__main__":
    img = 'people-racial-diversity.jpg'
    print(FaceAPI().detect(img).emotions())
