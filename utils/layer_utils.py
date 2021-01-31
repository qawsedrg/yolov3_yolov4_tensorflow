# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf

slim = tf.contrib.slim


def MISH(inputs):
    MISH_THRESH = 20.0
    tmp = inputs
    inputs = tf.where(
        tf.math.logical_and(tf.less(tmp, MISH_THRESH), tf.greater(inputs, -MISH_THRESH)),
        tf.log(1 + tf.exp(tmp)),
        tf.zeros_like(tmp)
    )
    inputs = tf.where(tf.less(tmp, -MISH_THRESH),
                      tf.exp(tmp),
                      inputs)
    inputs = tf.where(tf.greater(tmp, MISH_THRESH),
                      tmp,
                      inputs)
    inputs = tmp * tf.tanh(inputs)
    return inputs


def conv2d(inputs, filters, kernel_size=3, strides=1):
    def _fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs

    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs


def darknet53_body(inputs):
    def res_block(inputs, filters):
        shortcut = inputs
        net = conv2d(inputs, filters * 1, 1)
        net = conv2d(net, filters * 2, 3)
        net = net + shortcut
        return net

    net = conv2d(inputs, 32, 3, strides=1)
    net = conv2d(net, 64, 3, strides=2)
    net = res_block(net, 32)
    net = conv2d(net, 128, 3, strides=2)
    for i in range(2):
        net = res_block(net, 64)
    net = conv2d(net, 256, 3, strides=2)
    for i in range(8):
        net = res_block(net, 128)
    route_1 = net
    net = conv2d(net, 512, 3, strides=2)
    for i in range(8):
        net = res_block(net, 256)
    route_2 = net
    net = conv2d(net, 1024, 3, strides=2)
    for i in range(4):
        net = res_block(net, 512)
    route_3 = net
    return route_1, route_2, route_3


def yolo_block(inputs, filters):
    net = conv2d(inputs, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    route = net
    net = conv2d(net, filters * 2, 3)
    return route, net


def CSPdarknet53_body(inputs):
    def CSP(inputs, in_channels, res_num, double_ch=False):
        out_channels = in_channels
        if double_ch:
            out_channels = in_channels * 2

        net = conv2d(inputs, in_channels * 2, strides=2)
        route = conv2d(net, out_channels, kernel_size=1)
        net = conv2d(net, out_channels, kernel_size=1)

        for _ in range(res_num):
            tmp = net
            net = conv2d(net, in_channels, kernel_size=1)
            net = conv2d(net, out_channels)
            net = tmp + net

        net = conv2d(net, out_channels, kernel_size=1)
        net = tf.concat([net, route], -1)
        net = conv2d(net, in_channels * 2, kernel_size=1)

        return net

    net = conv2d(inputs, 32)
    net = CSP(net, 32, 1, double_ch=True)
    net = CSP(net, 64, 2)
    net = CSP(net, 128, 8)
    route_1 = net
    net = CSP(net, 256, 8)
    route_2 = net
    route_3 = CSP(net, 512, 4)

    return route_1, route_2, route_3


def yolo_blockv4(net, in_channels, a, b):
    for _ in range(a):
        out_channels = in_channels / 2
        net = conv2d(net, out_channels, kernel_size=1)
        net = conv2d(net, in_channels)

    out_channels = in_channels
    for _ in range(b):
        out_channels = out_channels / 2
        net = conv2d(net, out_channels, kernel_size=1)

    return net


def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')
    return inputs


def SSP(inputs):
    def yolo_maxpool_block(inputs):
        max_5 = tf.nn.max_pool(inputs, [1, 5, 5, 1], [1, 1, 1, 1], 'SAME')
        max_9 = tf.nn.max_pool(inputs, [1, 9, 9, 1], [1, 1, 1, 1], 'SAME')
        max_13 = tf.nn.max_pool(inputs, [1, 13, 13, 1], [1, 1, 1, 1], 'SAME')
        net = tf.concat([max_13, max_9, max_5, inputs], -1)
        return net

    net = yolo_blockv4(inputs, 1024, 1, 1)
    net = yolo_maxpool_block(net)
    net = yolo_blockv4(net, 1024, 1, 1)
    return net


def PANet(route_1, route_2, route_3, use_static_shape, class_num):
    route_up = conv2d(route_3, 256, kernel_size=1)
    route_up = upsample_layer(route_up, route_2.get_shape().as_list() if use_static_shape else tf.shape(route_2))
    route_2 = conv2d(route_2, 256, kernel_size=1)
    route_up = tf.concat([route_2, route_up], -1)
    route_up = yolo_blockv4(route_up, 512, 2, 1)
    route_middle = route_up
    route_up = conv2d(route_up, 128, kernel_size=1)
    route_up = upsample_layer(route_up, route_1.get_shape().as_list() if use_static_shape else tf.shape(route_1))
    route_1 = conv2d(route_1, 128, kernel_size=1)
    route_up = tf.concat([route_1, route_up], -1)
    net = yolo_blockv4(route_up, 256, 2, 1)
    route_up = conv2d(net, 256)
    route_up = slim.conv2d(route_up, 3 * (4 + 1 + class_num), 1,
                           stride=1, normalizer_fn=None,
                           activation_fn=None,
                           biases_initializer=tf.zeros_initializer())
    net = conv2d(net, 256, strides=2)
    route_middle = tf.concat([net, route_middle], -1)
    net = yolo_blockv4(route_middle, 512, 2, 1)
    route_middle = conv2d(net, 512)
    route_middle = slim.conv2d(route_middle, 3 * (4 + 1 + class_num), 1,
                               stride=1, normalizer_fn=None,
                               activation_fn=None,
                               biases_initializer=tf.zeros_initializer())
    net = conv2d(net, 512, strides=2)
    route_down = tf.concat([net, route_3], -1)
    route_down = yolo_blockv4(route_down, 1024, 2, 1)
    route_down = conv2d(route_down, 1024)
    route_down = slim.conv2d(route_down, 3 * (4 + 1 + class_num), 1,
                             stride=1, normalizer_fn=None,
                             activation_fn=None,
                             biases_initializer=tf.zeros_initializer())
    return route_up, route_middle, route_down


def yolo_Res_block(inputs, in_channels):
    net = conv2d(inputs, in_channels)
    route1 = net
    net = tf.split(net, num_or_size_splits=2, axis=-1)[1]
    net = conv2d(net, in_channels // 2)
    route2 = net
    net = conv2d(net, in_channels // 2)
    net = tf.concat([net, route2], axis=-1)
    net = conv2d(net, in_channels, kernel_size=1)
    return tf.concat([route1, net], axis=-1), net
