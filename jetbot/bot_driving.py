import cv2
import onnxruntime as rt

from pathlib import Path
import yaml
import numpy as np

from PUTDriver import PUTDriver, gstreamer_pipeline


def process(img):
    def region_of_interest(image):
        height, width = image.shape[:2]
        # Define the trapezoid vertices
        vertices = np.array([
            [(0, height),  # Bottom-left
             (width, height),  # Bottom-right
             (width * 0.6, height * 0.4),  # Top-right
             (width * 0.4, height * 0.4)]  # Top-left
        ], dtype=np.int32)
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, vertices, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 60, 100)
    dilated = cv2.dilate(canny, kernel=(5, 5))

    masked = region_of_interest(dilated)
    return masked.astype(np.float32) / 255.0

class AI:
    def __init__(self, config: dict):
        self.path = config['model']['path']

        self.sess = rt.InferenceSession(self.path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
 
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name

        self.history = [np.array([0,0], np.float32)] * 10

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        processed = process(img)
        processed[:90] = 0
        return processed[np.newaxis,...,np.newaxis]

    def postprocess(self, detections: np.ndarray) -> np.ndarray:
        self.history.append(np.array([0.8, 1.3 * detections[0][0]], dtype=np.float32))
        return self.history[-1]

    def predict(self, img: np.ndarray) -> np.ndarray:
        inputs = self.preprocess(img)

        assert inputs.dtype == np.float32
        
        detections = self.sess.run([self.output_name], {self.input_name: inputs})[0]
        outputs = self.postprocess(detections)

        assert outputs.dtype == np.float32

        return outputs


def main():
    with open("config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    driver = PUTDriver(config=config)
    ai = AI(config=config)

    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0, display_width=224, display_height=224), cv2.CAP_GSTREAMER)

    # model warm-up
    ret, image = video_capture.read()
    if not ret:
        print(f'No camera')
        return
    
    _ = ai.predict(image)

    input('Robot is ready to ride. Press Enter to start...')

    forward, left = 0.0, 0.0
    while True:
        print(f'Forward: {forward:.4f}\tLeft: {left:.4f}')
        driver.update(forward, left)

        ret, image = video_capture.read()
        if not ret:
            print(f'No camera')
            break
        forward, left = ai.predict(image)


if __name__ == '__main__':
    main()
