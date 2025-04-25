from typing import Dict, List, Tuple, Union
import cv2
import numpy as np
from datetime import datetime
import yaml
from pathlib import Path


def measure_time(func, *args, **kwargs):
    start_time = datetime.now()
    result = func(*args, **kwargs)
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    return result, total_time


class InferenceEngine:
    ENGINE = ["OpenCV", "OpenVINO", "Ultralytics", "MMdeploy", "CANN", "Hailo"]
    DEVICE = ["CPU", "NVIDIA GPU", "Intel NPU", "Ascend NPU", "Hailo-8"]

    def __init__(self,
                 config: Union[str, Path, Dict] = None,
                 weights_path: Union[str, Path] = None,
                 engine: str = 'Ultralytics',
                 device: str = 'CPU',
                 show: bool = False,
                 conf: float = 0.25,
                 iou: float = 0.45,
                 background: bool = 'Original',
                 show_classes: bool = False,
                 show_keypoints: bool = True,
                 show_skeletons: bool = False,
                 show_bbox: bool = True,
                 radius: int = 5,
                 skeleton_line_width: int = 2,
                 bbox_line_width: int = 2
                 ):
        self.model = None
        self._weights_path = weights_path
        self._engine = engine
        self._device = device
        self.visualize_config = {
            'conf': conf,
            'iou': iou,
           'show': show,
           'show_classes': show_classes,
           'show_keypoints': show_keypoints,
           'show_skeletons': show_skeletons,
           'show_bbox': show_bbox,
            'radius': radius,
           'skeleton_line_width': skeleton_line_width,
            'bbox_line_width': bbox_line_width,
            'background': background
        }
        self._data_config = None
        self._load_config(config)

    @property
    def engine(self):
        return self._engine

    @engine.setter
    def engine(self, engine):
        self._engine = engine

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    @property
    def weights_path(self):
        return self._weights_path

    @weights_path.setter
    def weights_path(self, weights_path):
        self._weights_path = weights_path

    @property
    def data_config(self):
        return self._data_config

    @data_config.setter
    def data_config(self, config):
        self._data_config = config
        self._update_config_vars()

    def _update_config_vars(self):
        self.classes = self.data_config.get('names', {})
        self.keypoints = self.data_config.get('kpt_shape', [])
        self.skeleton = self.data_config.get('skeleton', [])

        # Generate a color palette for the classes
        self.classes_color_palette = np.random.uniform(0, 255, size=(len(self.classes.keys()), 3))
        self.keypoints_color_palette = np.random.uniform(0, 255, size=(len(self.keypoints), 3))
        self.skeleton_color_palette = np.random.uniform(0, 255, size=(len(self.skeleton), 3))

    def update_config(self, config: Dict):
        for key, value in config.items():
            if key in self.visualize_config:
                self.visualize_config[key] = value
            else:
                raise ValueError(f"Invalid key {key} in config. Please choose from {self.visualize_config.keys()}")

    def _load_config(self, config: Union[str, Path, Dict] = None):

        if isinstance(config, str) or isinstance(config, Path):
            with open(config, 'r') as f:
                self._data_config = yaml.safe_load(f)
        elif isinstance(config, dict):
            self._data_config = config
        else:
            self._data_config = {
                'names': {},
                'kpt_shape': [],
               'skeleton': []
            }
        self._update_config_vars()

    def model_init(self):
        if self.engine not in self.ENGINE:
            raise ValueError(f"Engine {self.engine} is not supported. Please choose from {self.ENGINE}")
        if self.device not in self.DEVICE:
            raise ValueError(f"Device {self.device} is not supported. Please choose from {self.DEVICE}")

        if self.engine == 'Ultralytics':
            try:
                from ultralytics import YOLO
                self.model = YOLO(data=self.data_config,
                                  weights=self.weights_path,
                                  device=self.device)
            except ImportError:
                raise ImportError("Please install ultralytics to use Ultralytics engine.")
        elif self.engine == 'OpenCV':
            self.model = cv2.dnn.readNetFromONNX(self.weights_path)
        elif self.engine == 'OpenVINO':
            try:
                from openvino.inference_engine import IECore
                self.model = IECore().read_network(self.weights_path, self.data_config)
            except ImportError:
                raise ImportError("Please install openvino to use OpenVINO engine.")
        elif self.engine == 'MMdeploy':
            pass
        elif self.engine == 'CANN':
            try:
                from ais_bench.infer.interface import InferSession
                self.model = InferSession(0, self.weights_path)
            except ImportError:
                raise ImportError("Please install ais_bench to use CANN engine.")
        elif self.engine == 'Hailo':
            try:
                from hailo_model_zoo.core.infer.infer_utils import InferAPI
                self.model = InferAPI(self.weights_path, self.device)
            except ImportError:
                raise ImportError("Please install hailo_model_zoo to use Hailo engine.")

    def inference(self, frame):
        """
        Performs inference using an ONNX model and returns the
        output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """

        # Preprocess the image data
        [img, IM], preprocess_time = measure_time(self.preprocess, frame)

        # Run inference using the preprocessed image data
        if self.engine == 'Ultralytics':
            pred, inference_time = measure_time(self.model, [img], verbose=False)
        elif self.engine == 'OpenCV':
            pass
        elif self.engine == 'OpenVINO':
            pass
        elif self.engine == 'MMdeploy':
            pass
        elif self.engine == 'CANN':
            pred, inference_time = measure_time(self.model.infer, [img])
        elif self.engine == 'Hailo':
            pass
        # Perform post-processing on the outputs to obtain output image.
        results, postprecess_time = measure_time(self.postprocess, pred, IM)

        results['preprocess_time'] = preprocess_time
        results['inference_time'] = inference_time
        results['postprecess_time'] = postprecess_time
        results['fps'] = 1.0 / (inference_time + postprecess_time + preprocess_time + 1e-7)

        return results

    def preprocess(self, input_image):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        img = input_image.copy()

        # Get the height and width of the input image
        img_height, img_width = img.shape[:2]

        scale = min(self.input_width / img_width, self.input_height / img_height)

        ox = self.input_width - scale * img_width
        oy = self.input_height - scale * img_height

        M = np.array([
            [scale, 0, ox],
            [0, scale, oy],
        ], dtype="float32"
        )

        img = cv2.warpAffine(img, M,
                             (self.input_width, self.input_height),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(114, 114, 114))

        IM = cv2.invertAffineTransform(M)

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize the image data by dividing it by 255.0
        img = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        img = np.transpose(img, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        img = np.expand_dims(img, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return img, IM

    def postprocess(self, pred, IM=[]):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            pred (numpy.ndarray): The output of the model.
            1, 8400, boxes + classes + keypoints * 3 [cx, cy, w, h, conf1, ..., confn, nk*3]
            for example: for ap10k, it has 50 classes and 17 keypoints, so
            pred's shape is [1,8400, 4+50+17*3].
            IM (list): The list of input images to draw detections on.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(pred[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        keypoints_list = []
        scores = []
        class_ids = []

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:4 + len(self.classes)]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.conf:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = (x - w * 0.5) * IM[0][0] + IM[0][2]
                top = (y - h * 0.5) * IM[1][1] + IM[1][2]
                width = w * IM[0][0]
                height = h * IM[1][1]

                keypoints = outputs[i][4 + len(self.classes):].reshape(-1, 3)
                keypoints[:, 0] = keypoints[:, 0] * IM[0][0] + IM[0][2]
                keypoints[:, 1] = keypoints[:, 1] * IM[1][1] + IM[1][2]

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
                keypoints_list.append(keypoints)

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou)

        boxes = np.array(boxes)
        keypoints_list = np.array(keypoints_list)
        scores = np.array(scores)
        class_ids = np.array(class_ids)

        if indices is not None:
            indices = indices[0] if len(indices) == 1 else indices
            boxes = boxes[indices]
            keypoints_list = keypoints_list[indices]
            class_ids = class_ids[indices]
            scores = scores[indices]
            results = {
                'boxes': boxes,
                'keypoints_list': keypoints_list,
                'class_ids': class_ids,
               'scores': scores,
            }
        else:
            results = {
                'boxes': None,
                'keypoints_list': None,
                'class_ids': None,
               'scores': None,
            }
        # Return the modified input image
        return results

    def draw_detections(self, img, box, score, class_id, line_width=2):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.classes_color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(
            img=img,
            pt1=(int(x1), int(y1)),
            pt2=(int(x1 + w), int(y1 + h)),
            color=color,
            thickness=line_width,
            lineType=cv2.LINE_AA
        )

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = int(x1)
        label_y = int(y1 - 10) if y1 - 10 > label_height else int(y1 + 10)

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img,
            (label_x, label_y - label_height),
            (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1, cv2.LINE_AA)

    def draw_keypoints(self, img, keypoints, radius=5):
        for i, keypoint in enumerate(keypoints):
            x, y = keypoint[0], keypoint[1]
            color_k = [int(x) for x in self.keypoints_color_palette[i]]

            if x != 0 and y != 0:
                cv2.circle(img,
                           (int(x), int(y)),
                           radius=radius,
                           color=color_k,
                           thickness=-1,
                           lineType=cv2.LINE_AA)

    def draw_skeleton(self, img, keypoints, line_width=1):
        for i, sk in enumerate(self.skeleton):
            pos1 = (int(keypoints[(sk[0]), 0]), int(keypoints[(sk[0]), 1]))
            pos2 = (int(keypoints[(sk[1]), 0]), int(keypoints[(sk[1]), 1]))

            if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
                continue

            cv2.line(img,
                     pos1, pos2,
                     color=self.skeleton_color_palette[i],
                     thickness=line_width,
                     lineType=cv2.LINE_AA
                     )

    def process_frame(self, frame):
        """
        Processes a single frame of video and returns the output image with drawn detections.

        Args:
            frame: The input frame to process.

        Returns:
            output_img: The output image with drawn detections.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("Input frame must be a numpy array.")

        results = self.inference(frame)

        boxes = results['boxes']
        keypoints_list = results['keypoints_list']
        scores = results['scores']
        class_ids = results['class_ids']

        if self.visualize_config['background'] == 'Black':
            frame = np.zeros_like(frame)
        elif self.visualize_config['background'] == 'White':
            frame = np.ones_like(frame) * 255
        elif self.visualize_config['background'] == 'Original':
            pass

        if len(boxes) > 0:
            if boxes.ndim == 1:
                if self.visualize_config['show_bbox']:
                    self.draw_detections(frame, boxes, scores, class_ids)
                if self.visualize_config['show_keypoints']:
                    self.draw_keypoints(frame, keypoints_list,
                                        radius=self.visualize_config['radius'])
                if self.visualize_config['show_skeletons']:
                    self.draw_skeleton(frame, keypoints_list,
                                       line_width=self.visualize_config['skeleton_line_width'])
            else:
                for i, box in enumerate(boxes):
                    if self.visualize_config['show_bbox']:
                        self.draw_detections(frame, box, scores[i], class_ids[i])
                    if self.visualize_config['show_keypoints']:
                        self.draw_keypoints(frame, keypoints_list[i],
                                            radius=self.visualize_config['radius'])
                    if self.visualize_config['show_skeletons']:
                        self.draw_skeleton(frame, keypoints_list[i],
                                           line_width=self.visualize_config['skeleton_line_width'])

        times = {
            'preprocess_time': 'preprocess: {:.3f} ms'.format(results['preprocess_time'] * 1000.0),
            'inference_time': 'inference: {:.3f} ms'.format(results['inference_time'] * 1000.0),
            'postprocess_time': 'postprocess: {:.3f} ms'.format(results['postprecess_time'] * 1000.0),
            'fps': 'FPS: {:.1f} FPS'.format(results['fps'])
        }

        cv2.putText(frame, times['fps'], (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, times['preprocess_time'], (5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, times['inference_time'], (5, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, times['postprocess_time'], (5, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

        return frame, results
    