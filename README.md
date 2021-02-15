# A complete TensorFlow implementation of YOLOv3/4(Tiny)


--------

### 1. Introduction

This is a full implementation of YOLOv3, YOLOv4, YOLOv3-Tiny, YOLOv4-Tiny in pure TensorFlow. 

Key features:

-   K-means algorithm to select prior anchor boxes.

- Efficient tf.data multi-threading pipeline
- tensorlayer accelerated data augmentation, mixup, warm-up, label-smooth, focal-loss, multi-scale
- .weights to ckpt conversion
- Extremely fast GPU non maximum supression.
- ckpt to .pb and demo usage of .pb

### 2. Requirements

- tensorflow1(with tf.data support)
- opencv-python
- tqdm
- tensorlayer

### 3. Weights conversion

Weights can be downloaded here: 链接: https://pan.baidu.com/s/12Li_AZrZbGAs2642jjeOhw  密码: nkd2

Place the weights file in the correspond directory in `./data/` and run the correspond conversion script, the converted ckpt file will be saved to the save directory.

### 4. Data preparation

1.  annotation file

-   Generate `train.txt/val.txt` files under `./data/my_data/` directory. 

-   One line for one image, in the format like `index` `absolute_path` `width` `height` `box_1 box_2 ... box_n` separate with a white space. 

-   Box_x format: `label_index x_min y_min x_max y_max`.

-   `index` is the line index, starts from zero.

-   `label_index` is the index of label in `.names` file, starts from zero.

```
0 xxx/xxx/a.jpg 1920 1080 0 453 369 473 391 1 588 245 608 268
1 xxx/xxx/b.jpg 1920 1080 1 466 403 485 422 2 793 300 809 320
...
```

2.  class_name file:

-   Generate the `.names` file under `./data/` directory. Each line represents a class name.

```
bird
person
bike
...
```

The COCO dataset class names file is placed at `./data/coco.names`.

3.  anchor file:

-   Using the `get_kmeans.py` to get the prior anchors.

YOLO anchors is placed at `./data/yolo_anchors.txt`, YOLO-Tiny anchors is placed at `./data/yolo_tiny_anchors.txt`.

The yolo anchors computed by the k-means script is on the resized image scale.  The default resize method is the letterbox resize, i.e., keep the original aspect ratio in the resized image.

4.  VOC dataset:

    An example to parse VOC-like dataset is provided: `write_list_voc.py`, which can be used directly to parse VOCdevkit dataset.

### 5. Training

Run the correspond `train.py` file, continue training from a checkpoint is supported, you may refer to the tensorboard ouput in `./data/logs` .

Check the `args.py` for more details. You may set the parameters yourself in your own specific task.

### 6. Freeze graph

The correspond `freezegraph.py` can generate `.pb` model, with or without weights.

The method to use `.pb` file can found in the correspond `use_pb.py`. You may refer to `use_pb.py` without suffix 'gpu' to for method to decode the yolo output purely on cpu.

-------

### Credits:

I referred to these fantastic repositories:

[AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)

[wizyoung/YOLOv3_TensorFlow](https://github.com/wizyoung/YOLOv3_TensorFlow)

