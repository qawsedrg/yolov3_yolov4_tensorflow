import tensorflow as tf
from tqdm import trange

import argsv3 as args
from utils.data_utils import get_batch_data
from utils.eval_utils import get_preds_gpu, voc_eval, parse_gt_rec
from utils.misc_utils import parse_anchors, read_class_names, AverageMeter
from utils.nms_utils import gpu_nms

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))

from modelv3 import yolov3

# args params
anchors = parse_anchors(args.anchor_path)
classes = read_class_names(args.class_name_path)
class_num = len(args.classes)
img_cnt = len(open(args.val_file, 'r').readlines())

# setting placeholders
is_training = tf.placeholder(dtype=tf.bool, name="phase_train")
handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
gpu_nms_op = gpu_nms(pred_boxes_flag, pred_scores_flag, class_num, args.nms_topk, args.score_threshold,
                     args.nms_threshold)

##################
# tf.data pipeline
##################
val_dataset = tf.data.TextLineDataset(args.val_file)
val_dataset = val_dataset.batch(1)
val_dataset = val_dataset.map(
    lambda x: tf.py_func(get_batch_data,
                         [x, class_num, args.img_size, anchors, 'val', False, False, args.letterbox_resize],
                         [tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=args.num_threads
)
val_dataset.prefetch(args.prefetech_buffer)
iterator = val_dataset.make_one_shot_iterator()

image_ids, image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
image_ids.set_shape([None])
y_true = [y_true_13, y_true_26, y_true_52]
image.set_shape([None, args.img_size[1], args.img_size[0], 3])
for y in y_true:
    y.set_shape([None, None, None, None, None])

##################
# Model definition
##################
yolo_model = yolov3(class_num, anchors)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(image, is_training=is_training)
loss = yolo_model.compute_loss(pred_feature_maps, y_true)
y_pred = yolo_model.predict(pred_feature_maps)

saver_to_restore = tf.train.Saver()

with tf.Session(config=config) as sess:
    sess.run([tf.global_variables_initializer()])
    ckpt = tf.compat.v1.train.get_checkpoint_state(args.restore_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver_to_restore.restore(sess, ckpt.model_checkpoint_path)
        print("message: load ckpt model")
    else:
        print("message:can not fint ckpt model")

    print('\n----------- start to eval -----------\n')

    val_loss_total, val_loss_xy, val_loss_wh, val_loss_conf, val_loss_class = \
        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    val_preds = []

    for j in trange(img_cnt):
        __image_ids, __y_pred, __loss = sess.run([image_ids, y_pred, loss], feed_dict={is_training: False})
        pred_content = get_preds_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, __image_ids, __y_pred)

        val_preds.extend(pred_content)
        val_loss_total.update(__loss[0])
        val_loss_xy.update(__loss[1])
        val_loss_wh.update(__loss[2])
        val_loss_conf.update(__loss[3])
        val_loss_class.update(__loss[4])

    rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
    gt_dict = parse_gt_rec(args.val_file, args.img_size, args.letterbox_resize)
    print('mAP eval:')
    for ii in range(class_num):
        npos, nd, rec, prec, ap = voc_eval(gt_dict, val_preds, ii, iou_thres=0.5, use_07_metric=args.use_voc_07_metric)
        rec_total.update(rec, npos)
        prec_total.update(prec, nd)
        ap_total.update(ap, 1)
        print('Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}'.format(ii, rec, prec, ap))

    mAP = ap_total.average
    print('final mAP: {:.4f}'.format(mAP))
    print("recall: {:.3f}, precision: {:.3f}".format(rec_total.average, prec_total.average))
    print("total_loss: {:.3f}, loss_xy: {:.3f}, loss_wh: {:.3f}, loss_conf: {:.3f}, loss_class: {:.3f}".format(
        val_loss_total.average, val_loss_xy.average, val_loss_wh.average, val_loss_conf.average, val_loss_class.average
    ))
