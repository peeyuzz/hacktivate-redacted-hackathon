import pytesseract
from PIL import Image
import os
import cv2
from openvino.runtime import Core
import numpy as np
import utils
import platform


# Define tesseract command path based on the OS
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
elif platform.system() == "Linux":
    # Typically Tesseract is installed in /usr/bin/tesseract on Linux
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

class FaceDetector:
    def __init__(self,
                 model,
                 confidence_thr=0.3,
                 overlap_thr=0.7):
        # load and compile the model
        core = Core()
        model = core.read_model(model=model)
        compiled_model = core.compile_model(model=model)
        self.model = compiled_model

        # 'Cause that model has more than one output,
        # We are saving the names in a more human frendly
        # variable to remember later how to recover the output we wish
        # In our case here, is a output for hte bbox and other for the score
        # /confidence. Have a look at the openvino documentation for more i
        self.output_scores_layer = self.model.output(0)
        self.output_boxes_layer = self.model.output(1)
        # confidence threshold
        self.confidence_thr = confidence_thr
        # threshold for the nonmaximum suppression
        self.overlap_thr = overlap_thr

    def preprocess(self, image):
        """
            input image is a numpy array image representation,
            in the BGR format of any shape.
        """
        # resize to match the expected by the model
        input_image = cv2.resize(image, dsize=[320, 240])
        # changing from [H, W, C] to [C, H, W]. "channels first"
        input_image = np.expand_dims(input_image.transpose(2, 0, 1), axis=0)
        return input_image

    def posprocess(self, pred_scores, pred_boxes, image_shape):
        # get all predictions with more than confidence_thr of confidence
        filtered_indexes = np.argwhere(
            pred_scores[0, :, 1] > self.confidence_thr).tolist()
        filtered_boxes = pred_boxes[0, filtered_indexes, :]
        filtered_scores = pred_scores[0, filtered_indexes, 1]

        if len(filtered_scores) == 0:
            return [], []

        # convert all boxes to image coordinates
        h, w = image_shape

        def _convert_bbox_format(*args):
            bbox = args[0]
            x_min, y_min, x_max, y_max = bbox
            x_min = int(w*x_min)
            y_min = int(h*y_min)
            x_max = int(w*x_max)
            y_max = int(h*y_max)
            return x_min, y_min, x_max, y_max

        bboxes_image_coord = np.apply_along_axis(
            _convert_bbox_format, axis=2, arr=filtered_boxes)

        # apply non-maximum supressions
        bboxes_image_coord, indexes = utils.non_max_suppression(bboxes_image_coord.reshape([-1, 4]),
                                                                overlapThresh=self.overlap_thr)
        filtered_scores = filtered_scores[indexes]
        return bboxes_image_coord, filtered_scores

    def draw_bboxes(self, image, bboxes, color=[0, 255, 0]):
        # Just for visualization
        # draw all bboxes on the input image
        for boxe in bboxes:
            x_min, y_min, x_max, y_max = boxe
            pt1 = (x_min, y_min)
            pt2 = (x_max, y_max)
            cv2.rectangle(image, pt1, pt2, color=color,
                          thickness=2, lineType=cv2.LINE_4)  # BGR

    def inference(self, image):
        input_image = self.preprocess(image)
        # inference
        pred_scores = self.model([input_image])[self.output_scores_layer]
        pred_boxes = self.model([input_image])[self.output_boxes_layer]

        image_shape = image.shape[:2]
        faces, scores = self.posprocess(pred_scores, pred_boxes, image_shape)
        return faces, scores
    

    ##########################################################################
