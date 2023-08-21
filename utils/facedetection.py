import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
from keras import backend as K
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf

from numpy import expand_dims
from keras.models import load_model, Model
from keras.utils import load_img, img_to_array
from keras.models import Model, Sequential
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from django.core.files.storage import default_storage
from attendancesystems.settings import BASE_DIR, MEDIA_ROOT
from django.core.files.storage import FileSystemStorage

# parameters for models
detector_w, detector_h = 416, 416
encoder_w, encoder_h = 224, 224

class_threshold = 0.6
labels = ["face"]
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]  

# loading models
detection_model = load_model(BASE_DIR.joinpath('utils/models/model.h5'), compile=False)
model = load_model(BASE_DIR.joinpath('utils/models/vgg_face_similiarity.h5'), compile=False)
encoder_model = Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1
        
    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        
        return self.score


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def detector_image_preprocess(image, target_size=(detector_w, detector_h)):

    image = cv2.resize(image, target_size)
    image = image.astype('float32') 
    image /= 255.0  
    image = expand_dims(image, 0)
    return image

def encoder_image_preprocess(img, target_size=(224,224)):
    img = cv2.resize(img, target_size)
    img = expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2] # 0 and 1 is row and column 13*13
    nb_box = 3 # 3 anchor boxes
    netout = netout.reshape((grid_h, grid_w, nb_box, -1)) #13*13*3 ,-1
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh
    
    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if(objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            boxes.append(box)
    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3


#intersection over union        
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    
    
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin  
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    #Union(A,B) = A + B - Inter(A,B)
    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union

def do_non_maximum_supression(boxes, nms_thresh):    
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    for box in boxes:
        for i in range(len(labels)):
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
    
    return v_boxes, v_labels, v_scores

# draw all results
def draw_boxes(img, v_boxes, v_labels, v_scores):
    color = (0, 0, 255) 
    thickness = 2
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        start_point = (x1, y1) 
        end_point = (x2, y2) 
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
        
    return img


def find_cosine_similarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation),test_representation)
    b = np.sum(np.multiply(source_representation,source_representation))
    c = np.sum(np.multiply(test_representation,test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def get_face_encodings(file, save_path=None, is_attendance=True):
    image = Image.open(file, mode='r')
    image_w, image_h = image.size
    image = img_to_array(image)
  

    img = detector_image_preprocess(image)
    yhat = detection_model.predict(img)

    boxes = list()
    for i in range(len(yhat)):
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, detector_h, detector_w)
  
    correct_yolo_boxes(boxes, image_h, image_w, detector_h, detector_w)
    do_non_maximum_supression(boxes, 0.5)
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

    if is_attendance:
        img_copy = image.copy()
        annotated_img = draw_boxes(img_copy, v_boxes, v_labels, v_scores)
        file_loc = os.path.join(save_path, 'annotated_faces.jpg')
        cv2.imwrite(file_loc, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        #file_name = default_storage.save('marked.jpg', img)

        face_save_loc = os.path.join(save_path, 'faces')
        try:
            os.mkdir(face_save_loc)
        except:
            pass
    

    face_encodings = list()
    for i in range(len(v_boxes)):
        img_copy = image.copy()
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        face = img_copy[y1:y2, x1:x2]
        if is_attendance:
            face_name = f"face_{i}.jpg"
            cv2.imwrite(os.path.join(face_save_loc, face_name), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

        face_processed = encoder_image_preprocess(face)
        vect = encoder_model.predict(face_processed)[0,:]
        face_encodings.append(vect)

    return face_encodings

def get_minimum_similiarity(student_face_vector, classroom_face_vectors):
    scores = list()
    for vec in classroom_face_vectors:
        score = find_cosine_similarity(student_face_vector, vec)
        scores.append(score)
    #print(scores)
    #print(f'Matched face index:{scores.index(min(scores))}, with similiarity: {min(scores)}')
    return scores
