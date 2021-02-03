from utils.layer_utils import MISH
import tensorflow as tf
slim=tf.contrib.slim
class Layer(object):
    def __init__(self, class_num, anchors=None, use_label_smooth=False, use_focal_loss=False, batch_norm_decay=0.999,
                 weight_decay=5e-4, use_static_shape=True,is_training=False):
        # self.anchors = [[10, 13], [16, 30], [33, 23],
        # [30, 61], [62, 45], [59,  119],
        # [116, 90], [156, 198], [373,326]]
        self.class_num = class_num
        self.anchors = anchors
        self.batch_norm_decay = batch_norm_decay
        self.use_label_smooth = use_label_smooth
        self.use_focal_loss = use_focal_loss
        self.weight_decay = weight_decay
        # inference speed optimization
        # if `use_static_shape` is True, use tensor.get_shape(), otherwise use tf.shape(tensor)
        # static_shape is slightly faster
        self.use_static_shape = use_static_shape
        self.batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }
    def set_img_size(self, img_size):
        self.img_size = img_size

    def get_batch_norm_params(self):
        return self.batch_norm_params

class Activation(object):
    def __init__(self,act):
        self.activation=act
    def act(self,x):
        if self.activation=="linear":
            return x
        if self.activation == "mish":
            return MISH(x)
        if self.activation=="leaky":
            return tf.nn.leaky_relu(x, alpha=0.1)
    def __str__(self):
        if self.activation=="linear":
            return "lambda x : {}".format("x")
        if self.activation == "mish":
            return "lambda x : {}".format("MISH(x)")
        if self.activation=="leaky":
            return "lambda x : {}".format("tf.nn.leaky_relu(x, alpha=0.1)")