import cv2
import numpy as np
import onnxruntime as rt
import yaml
from PIL import Image
from PUTDriver import PUTDriver, gstreamer_pipeline
from torchvision import transforms


class AI:
    def __init__(self, config: dict):
        self.path = config["model"]["path"]

        self.sess = rt.InferenceSession(
            self.path,
            providers=[
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        return np.array(preprocess(Image.fromarray(img)))[np.newaxis].astype(np.float32)

    def postprocess(self, detections: np.ndarray) -> np.ndarray:
        return detections.astype(np.float32)[0]

    def predict(self, img: np.ndarray) -> np.ndarray:
        inputs = self.preprocess(img)

        assert inputs.dtype == np.float32
        assert inputs.shape == (1, 3, 224, 224)

        detections = self.sess.run([self.output_name], {self.input_name: inputs})[0]
        outputs = self.postprocess(detections)

        assert outputs.dtype == np.float32
        assert outputs.shape == (2,)
        assert outputs.max() < 1.0
        assert outputs.min() > -1.0

        return outputs


def main():
    with open("config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    driver = PUTDriver(config=config)
    ai = AI(config=config)

    video_capture = cv2.VideoCapture(
        gstreamer_pipeline(flip_method=0, display_width=224, display_height=224),
        cv2.CAP_GSTREAMER,
    )

    # model warm-up
    ret, image = video_capture.read()
    if not ret:
        print("No camera")
        return

    _ = ai.predict(image)

    input("Robot is ready to ride. Press Enter to start...")

    forward, left = 0.0, 0.0
    while True:
        print(f"Forward: {forward:.4f}\tLeft: {left:.4f}")
        driver.update(forward, left)

        ret, image = video_capture.read()
        if not ret:
            print("No camera")
            break
        forward, left = ai.predict(image)


if __name__ == "__main__":
    main()
