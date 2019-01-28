# -*- coding: utf-8 -*-
# test-detection.py
import numpy as np
import cv2
import time
import os
import mss
import keyboard_action
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class TOD(object):
    def __init__(self):
        self.PATH_TO_CKPT 	= '/home/tianyiliu/Documents/workspace/gaming/myprojects/renderBench/modelData/supertuxkart/models/frozen_inference_graph.pb'
        self.PATH_TO_LABELS 	= '/home/tianyiliu/Documents/workspace/gaming/myprojects/renderBench/modelData/supertuxkart/label_map.pbtxt'
        self.NUM_CLASSES 	= 1
        self.detection_graph 	= self._load_model()
        self.category_index 	= self._load_label_map()
        self.shape   = [100,670,112,580,70,540,70,420,115,416,165,461,182,505,180,585]
        self.last_box_center   = [100,680]

    def _load_model(self):
        detection_graph 	= tf.Graph()
        with detection_graph.as_default():
            od_graph_def	= tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph



    def _load_label_map(self):
        label_map	= label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories	= label_map_util.convert_label_map_to_categories(label_map,max_num_classes=self.NUM_CLASSES,use_display_name=True)
        category_index	= label_map_util.create_category_index(categories)
        return category_index



    def detect(self, image):

        with self.detection_graph.as_default():

            with tf.Session(graph=self.detection_graph) as sess:

                image_np_expanded = np.expand_dims(image, axis=0)

                image_tensor	  = self.detection_graph.get_tensor_by_name('image_tensor:0')

                boxes		  = self.detection_graph.get_tensor_by_name('detection_boxes:0')

                scores		  = self.detection_graph.get_tensor_by_name('detection_scores:0')

                classes		  = self.detection_graph.get_tensor_by_name('detection_classes:0')

                num_detections	  = self.detection_graph.get_tensor_by_name('num_detections:0')



                (boxes, scores, classes, num_detections) = sess.run(

                       [boxes, scores, classes, num_detections],

                       feed_dict={image_tensor: image_np_expanded})

                #print(num_detections)

                box_centers=vis_util.visualize_realboxes_and_labels_on_image_array(

                       image,

                       np.squeeze(boxes),

                       np.squeeze(classes).astype(np.int32),\

                       np.squeeze(scores),

                       self.category_index,

                       use_normalized_coordinates=True,

                       line_thickness = 8)

                if len(box_centers) <= 0:

                    box_centers = [self.last_box_center[0]+1,self.last_box_center[1]+1]

                    feature_vector = self.shape + self.last_box_center + box_centers

                    self.last_box_center = box_centers

                else:

                    feature_vector = self.shape + self.last_box_center + box_centers[0]

                    self.last_box_center = box_centers[0]



        cv2.namedWindow("detection", cv2.WINDOW_NORMAL)

        cv2.imshow("detection", image)

        return feature_vector

if __name__ == '__main__':

    file_name = 'rnn6-x/raw-data/training_data' + str(int(time.time())) + '.npy'

    if os.path.isfile(file_name):

        print('File exists, loading previous data!')

        training_data = list(np.load(file_name))

    else:

        print('File does not exist, starting fresh!')

        training_data = []



    for i in list(range(4))[::-1]:

        print(i+1)

        time.sleep(1)




    with mss.mss(display=':0.0') as sct:
        region={'top': 65, 'left': 64, 'width': 1020, 'height': 750}
        detector = TOD()
        while(True):
            last_time = time.time()
            print('Frame took {} seconds'.format(time.time()-last_time))
            screen = np.array(sct.grab(region))
            output_keys 	= keyboard_action.check_keys() #only one hot key:[1,0,0] or [0,1,0] or [0,0,1]
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            figure_vector 	= [detector.detect(screen),output_keys]
            print(figure_vector)
            training_data.append(figure_vector)

            if len(training_data) % 100 == 0:
                print(len(training_data))
                np.save(file_name, training_data)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindow()
                break

