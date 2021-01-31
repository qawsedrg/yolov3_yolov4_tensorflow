import cv2

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from modelv3 import yolov3
from utils.data_aug import letterbox_resize
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))

anchor_path = "./data/yolo_anchors.txt"
class_name_path = "./data/coco.names"
anchors = parse_anchors(anchor_path)
classes = read_class_names(class_name_path)
class_num = len(classes)
color_table = get_color_table(class_num)
img_size = [416, 416]
yolo_model = yolov3(class_num, anchors)
yolo_model.set_img_size(np.asarray(img_size))

sess = tf.Session(config=config)
with gfile.FastGFile("./pb_model/frozen_model_v3.pb",'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
sess.run(tf.global_variables_initializer())
input = sess.graph.get_tensor_by_name('image:0')
feature_map_1 = sess.graph.get_tensor_by_name('yolov3/yolov3_head/feature_map_1:0')
feature_map_2 = sess.graph.get_tensor_by_name('yolov3/yolov3_head/feature_map_2:0')
feature_map_3 = sess.graph.get_tensor_by_name('yolov3/yolov3_head/feature_map_3:0')
pred_boxes, pred_confs, pred_probs = yolo_model.predict([feature_map_1, feature_map_2, feature_map_3])
pred_scores = pred_confs * pred_probs
boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, class_num, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

img_ori = cv2.imread("./data/demo_data/dog.jpg")
img, resize_ratio, dw, dh = letterbox_resize(img_ori, img_size[0], img_size[1])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.asarray(img, np.float32)
img = img[np.newaxis, :] / 255.

boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input: img})
boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio

for i in range(len(boxes_)):
    x0, y0, x1, y1 = boxes_[i]
    plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100),
                 color=color_table[labels_[i]])
cv2.imshow("img", img_ori)
cv2.waitKey(0)
