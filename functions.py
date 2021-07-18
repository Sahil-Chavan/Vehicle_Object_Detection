import os
import cv2
import time
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from tensorflow.keras.preprocessing.image import img_to_array,load_img

from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


def load_model():

    PATH_TO_SAVED_MODEL = "exported-models/my_ssd_mobilenet_v2_320x320/saved_model"
    print('Loading Model...')
    start_time = time.time()
    loaded_model = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    
    print('Done! Took {} seconds'.format(time.time() - start_time))
    
    return loaded_model

def show_inference(model,test_image_rgb):
    PATH_TO_LABELS = "annotations/label_map.pbtxt"
    # Loading the pbtxt
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)

    # test_image_o = cv2.imread(img_path)
    # test_image_rgb = cv2.cvtColor(test_image_o, cv2.COLOR_BGR2RGB)
    test_image_array_ex = np.expand_dims(test_image_rgb, axis=0)

    test_image_tensor = tf.convert_to_tensor(test_image_array_ex)
    test_image_tensor = tf.cast(test_image_tensor, tf.uint8) #changinig the dtype according to serving_default.

    # Predictions
    prediction = model(test_image_tensor)

    num_bboxes = int(prediction.pop('num_detections'))
    prediction = {key: value[0, :num_bboxes].numpy() for key, value in prediction.items()}
    prediction['num_detections'] = num_bboxes

    # detection_classes should be ints.
    prediction['detection_classes'] = prediction['detection_classes'].astype(np.int64)

    image_with_detections = test_image_rgb.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_with_detections,
          prediction['detection_boxes'],
          prediction['detection_classes'],
          prediction['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=5,
          min_score_thresh=0.4,
          agnostic_mode=False)

    return image_with_detections
