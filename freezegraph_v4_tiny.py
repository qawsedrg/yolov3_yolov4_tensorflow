import tensorflow as tf
from tensorflow.python.tools import freeze_graph

from modelv4tiny import yolov4tiny
from utils.misc_utils import parse_anchors, read_class_names

anchor_path = "./data/yolo_tiny_anchors.txt"
new_size = [416, 416]
class_name_path = "./data/coco.names"
anchors = parse_anchors(anchor_path)
classes = read_class_names(class_name_path)
num_class = len(classes)
yolo_model = yolov4tiny(num_class, anchors)


def main():
    tf.reset_default_graph()

    image = tf.placeholder(tf.float32, shape=(1, new_size[0], new_size[1], 3), name="image")
    with tf.variable_scope('yolov4tiny'):
        feature_map_1, feature_map_2 = yolo_model.forward(image, is_training=False)

    yolov4tiny_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolov4tiny')
    saver = tf.train.Saver(yolov4tiny_vars)
    with tf.Session() as sess:

        ckpt = tf.compat.v1.train.get_checkpoint_state("./data/darknet_weights_v4tiny/")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("message: load ckpt model")
        else:
            print("message:can not fint ckpt model")

        # 保存图
        tf.train.write_graph(sess.graph_def, './pb_model/', 'model_v4tiny.pb')
        # 把图和参数结构一起
        freeze_graph.freeze_graph('./pb_model/model_v4tiny.pb', '', False, ckpt.model_checkpoint_path,
                                  'yolov4tiny/head/feature_map_1, yolov4tiny/head/feature_map_2',
                                  'save/restore_all', 'save/Const:0', './pb_model/frozen_model_v4tiny.pb', False, "")
    print("done")


if __name__ == '__main__':
    main()
