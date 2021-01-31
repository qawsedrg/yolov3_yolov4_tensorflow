import tensorflow as tf

from modelv4tiny import yolov4tiny
from utils.misc_utils import parse_anchors, load_weights

num_class = 80
img_size = 416
weight_path = './data/darknet_weights_v4tiny/yolov4-tiny.weights'
save_path = './data/darknet_weights_v4tiny/yolov4-tiny.ckpt'
anchors = parse_anchors('./data/yolo_tiny_anchors.txt')

model = yolov4tiny(80, anchors)
with tf.Session() as sess:
    inputs = tf.placeholder(tf.float32, [1, img_size, img_size, 3])

    with tf.variable_scope('yolov4tiny'):
        feature_map = model.forward(inputs)

    saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov4tiny'))

    load_ops = load_weights(tf.global_variables(scope='yolov4tiny'), weight_path)
    sess.run(load_ops)
    saver.save(sess, save_path=save_path)
    print('TensorFlow model checkpoint has been saved to {}'.format(save_path))
