import cv2
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))


def read_class_names(class_name_path):
    names = {}
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    """
    Pure Python NMS baseline.

    Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
                      exact number of boxes
               scores: shape of [-1,]
               max_boxes: representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh: representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep[:max_boxes]


def cpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    """
    Perform NMS on CPU.
    Arguments:
        boxes: shape [1, 10647, 4]
        scores: shape [1, 10647, num_classes]
    """

    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1, num_classes)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []

    for i in range(num_classes):
        indices = np.where(scores[:, i] >= score_thresh)
        filter_boxes = boxes[indices]
        filter_scores = scores[:, i][indices]
        if len(filter_boxes) == 0:
            continue
        # do non_max_suppression on the cpu
        indices = py_nms(filter_boxes, filter_scores,
                         max_boxes=max_boxes, iou_thresh=iou_thresh)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32') * i)
    if len(picked_boxes) == 0:
        return None, None, None

    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)

    return boxes, score, label


def letterbox_resize(img, new_width, new_height, interp=0):
    '''
    Letterbox resize. keep the original aspect ratio in the resized image.
    '''
    ori_height, ori_width = img.shape[:2]

    resize_ratio = min(new_width / ori_width, new_height / ori_height)

    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)

    img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)
    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img

    return image_padded, resize_ratio, dw, dh


def get_color_table(class_num, seed=4):
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table


def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)


def reorg_layer_numpy(feature_map, anchors):
    '''
    feature_map: a feature_map from [feature_map_1, feature_map_2] returned
        from `forward` function
    anchors: shape: [3, 2]
    '''
    # NOTE: size in [h, w] format! don't get messed up!
    grid_size = feature_map.shape[1:3]  # [13, 13]
    # the downscale ratio in height and weight
    ratio = img_size / grid_size
    ratio.astype(np.float32)
    # rescale the anchors to the feature_map
    # NOTE: the anchor is in [w, h] format!
    rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

    feature_map = np.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + class_num])

    # split the feature_map along the last dimension
    # shape info: take 416x416 input image and the 13*13 feature_map for example:
    # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
    # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
    # conf_logits: [N, 13, 13, 3, 1]
    # prob_logits: [N, 13, 13, 3, class_num]
    box_centers, box_sizes, conf_logits, prob_logits = np.split(feature_map, [2, 4, 5], axis=-1)

    box_centers = sigmoid(box_centers)

    # use some broadcast tricks to get the mesh coordinates
    grid_x = np.linspace(0, grid_size[1], grid_size[1], dtype=np.int32)
    grid_y = np.linspace(0, grid_size[0], grid_size[0], dtype=np.int32)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    x_offset = np.reshape(grid_x, (-1, 1))
    y_offset = np.reshape(grid_y, (-1, 1))
    x_y_offset = np.concatenate([x_offset, y_offset], axis=-1)
    # shape: [13, 13, 1, 2]
    x_y_offset = np.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2])
    x_y_offset.astype(np.float32)

    # get the absolute box coordinates on the feature_map
    box_centers = box_centers + x_y_offset
    # rescale to the original image scale
    box_centers = box_centers * ratio[::-1]

    # avoid getting possible nan value with tf.clip_by_value
    box_sizes = np.exp(box_sizes) * rescaled_anchors
    # box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 100) * rescaled_anchors
    # rescale to the original image scale
    box_sizes = box_sizes * ratio[::-1]

    # shape: [N, 13, 13, 3, 4]
    # last dimension: (center_x, center_y, w, h)
    boxes = np.concatenate([box_centers, box_sizes], axis=-1)

    # shape:
    # x_y_offset: [13, 13, 1, 2]
    # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
    # conf_logits: [N, 13, 13, 3, 1]
    # prob_logits: [N, 13, 13, 3, class_num]
    return x_y_offset, boxes, conf_logits, prob_logits


def predict_numpy(feature_maps):
    feature_map_1, feature_map_2 = feature_maps

    feature_map_anchors = [(feature_map_1, anchors[3:6]),
                           (feature_map_2, anchors[0:3])]
    reorg_results = [reorg_layer_numpy(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

    def _reshape(result):
        x_y_offset, boxes, conf_logits, prob_logits = result
        grid_size = x_y_offset.shape[:2]
        boxes = np.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
        conf_logits = np.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
        prob_logits = np.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, class_num])
        # shape: (take 416*416 input image and feature_map_1 for example)
        # boxes: [N, 13*13*3, 4]
        # conf_logits: [N, 13*13*3, 1]
        # prob_logits: [N, 13*13*3, class_num]
        return boxes, conf_logits, prob_logits

    boxes_list, confs_list, probs_list = [], [], []
    for result in reorg_results:
        boxes, conf_logits, prob_logits = _reshape(result)
        confs = sigmoid(conf_logits)
        probs = sigmoid(prob_logits)
        boxes_list.append(boxes)
        confs_list.append(confs)
        probs_list.append(probs)

    # collect results on three scales
    # take 416*416 input image for example:
    # shape: [N, (13*13+26*26+52*52)*3, 4]
    boxes = np.concatenate(boxes_list, axis=1)
    # shape: [N, (13*13+26*26+52*52)*3, 1]
    confs = np.concatenate(confs_list, axis=1)
    # shape: [N, (13*13+26*26+52*52)*3, class_num]
    probs = np.concatenate(probs_list, axis=1)

    center_x, center_y, width, height = np.split(boxes, [1, 2, 3], axis=-1)
    x_min = center_x - width / 2
    y_min = center_y - height / 2
    x_max = center_x + width / 2
    y_max = center_y + height / 2

    boxes = np.concatenate([x_min, y_min, x_max, y_max], axis=-1)

    return boxes, confs, probs


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

anchor_path = "./data/yolo_tiny_anchors.txt"
class_name_path = "./data/coco.names"
anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])
classes = read_class_names(class_name_path)
class_num = len(classes)
color_table = get_color_table(class_num)
img_size = np.asarray([416, 416])

sess = tf.Session(config=config)
with gfile.FastGFile("./pb_model/frozen_model_v4tiny.pb",
                     'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
sess.run(tf.global_variables_initializer())
input = sess.graph.get_tensor_by_name('image:0')
feature_map_1 = sess.graph.get_tensor_by_name('yolov4tiny/head/feature_map_1:0')
feature_map_2 = sess.graph.get_tensor_by_name('yolov4tiny/head/feature_map_2:0')

#preprocess image
img_ori = cv2.imread("./data/demo_data/kite.jpg")
img, resize_ratio, dw, dh = letterbox_resize(img_ori, img_size[0], img_size[1])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.asarray(img, np.float32)
img = img[np.newaxis, :] / 255.

#inference
feature_1, feature_2 = sess.run([feature_map_1, feature_map_2], feed_dict={input: img})
#decode
pred_boxes, pred_confs, pred_probs = predict_numpy([feature_1, feature_2])
pred_scores = pred_confs * pred_probs
boxes, scores, labels = cpu_nms(pred_boxes, pred_scores, class_num, max_boxes=200, score_thresh=0.3, iou_thresh=0.45)
boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / resize_ratio
boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / resize_ratio

for i in range(len(boxes)):
    x0, y0, x1, y1 = boxes[i]
    plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels[i]] + ', {:.2f}%'.format(scores[i] * 100),
                 color=color_table[labels[i]])
cv2.imshow("img", img_ori)
cv2.waitKey(0)
