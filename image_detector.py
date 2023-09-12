import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()

model_file = os.getenv('MODEL_FILE')

base_options = python.BaseOptions(model_asset_path=model_file)
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)




def annotate_image(image, detection_result):
    MARGIN = 10
    ROW_SIZE = 10
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    TEXT_COLOR = (255, 0, 0)
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f'{category_name} {probability}'
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)

        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def detect_image(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    np_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    results = detector.detect(np_image)
    return results


def object_detection(source):
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        results = detect_image(image)

        annotated_image = annotate_image(image, results)


        yield annotated_image








