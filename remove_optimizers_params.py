import os
import tensorflow as tf
from modelv3 import yolov3

# params
restore_path = './data/darknet_weights_v3/'
class_num = 80
save_dir = './ckpt_without_optimizer/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

image = tf.placeholder(tf.float32, [1, 416, 416, 3])
yolo_model = yolov3(class_num, None)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(image)

saver_to_restore = tf.train.Saver()
saver_to_save = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.compat.v1.train.get_checkpoint_state(restore_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver_to_restore.restore(sess, ckpt.model_checkpoint_path)
        saver_to_save.save(sess, save_dir + ckpt.model_checkpoint_path.split('/')[-1])
        print("done")
    else:
        print("message:can not fint ckpt model")