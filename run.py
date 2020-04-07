# # Import packages
import os
import cv2
import sys
import time
import numpy as np
import pytesseract
import pandas as pd
from PIL import Image
from io import StringIO
import tensorflow as tf
from flask_cors import CORS, cross_origin
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify

# Import from Tensorflow object detection API
from utils import label_map_util
from utils import visualization_utils as vis_util

#Import from custom plate folder
from custom import preprocessing
from custom import character

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'numplate'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# Path to label map file
PATH_TO_LABELS = os.path.join('training', 'labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Create flask application
app = Flask(__name__)

# Number plate recognition
@app.route('/vehicle', methods= ['POST'])
def get_image():
    image = request.files["image"]
    image = Image.open(image) 
    imagearray = preprocessing.imageToarray(image)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            image_np_expanded = np.expand_dims(imagearray, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            ymin = boxes[0,0,0] 
            xmin = boxes[0,0,1]
            ymax = boxes[0,0,2]
            xmax = boxes[0,0,3]
            (im_width, im_height) = image.size
            (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
            cropped_image = tf.image.crop_to_bounding_box(imagearray, int(yminn), int(xminn),int(ymaxx - yminn), int(xmaxx - xminn))
            imgData = sess.run(cropped_image) 
            count = 0
            filename = preprocessing.croppedImage(imgData, count)
            pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\tesseract.exe'
            text = pytesseract.image_to_string(Image.open(filename),lang=None)
            print('CHARCTER RECOGNITION : ',character.catch_rectify_plate_characters(text))
            return jsonify({"output": text})
            # Draw the results of the detection (aka 'visulaize the results')
            vis_util.visualize_boxes_and_labels_on_image_array(
                imagearray,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.60)
            
if __name__ == "__main__": 
    app.run(host='0.0.0.0', port=5007)