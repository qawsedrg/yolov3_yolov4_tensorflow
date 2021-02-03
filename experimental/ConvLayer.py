from Layer import *


class ConvLayer():
    def __init__(self, Layer, ptr, **kwargs):
        self.batch_normalize = int(kwargs.get("batch_normalize", 0))
        self.filters = int(kwargs.get("filters", -1))
        self.size = int(kwargs.get("size", -1))
        self.stride = int(kwargs.get("stride", -1))
        self.pad = int(kwargs.get("pad", -1))
        self.activation = kwargs.get("activation", "-1")
        self.act = Activation(self.activation)
        self.normalizer_fn = slim.batch_norm if self.batch_normalize == 1 else None
        self.normalizer_params = Layer.batch_norm_params if self.batch_normalize == 1 else None
        self.ptr = ptr
        self.yolo=False
        self.class_num=Layer.class_num
        # self.output=slim.conv2d(self.inputs, self.filters, self.size, stride=self.stride, normalizer_fn=self.normalizer_fn,
        #                   normalizer_params=self.batch_norm_params, activation_fn=lambda x: self.act.act(x))

    @staticmethod
    def _fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs

    def set_input(self, inputs,):
        self.inputs = inputs
    def before_yolo(self,feature_map_num):
        self.yolo=True
        self.feature_map_num=feature_map_num
        assert self.filters==3*(5+self.class_num)

    def __str__(self):
        if self.yolo:
            return "\t\tx{:} = slim.conv2d({:}, {:}, {:}, stride={:}, normalizer_fn={:}, normalizer_params={:}, activation_fn={:},biases_initializer=tf.zeros_initializer())\n\t\tfeature_map_{:} = tf.identity(x{:}, name='feature_map_{}')".format(
                self.ptr, self.inputs, self.filters, self.size, self.stride,
                "slim.batch_norm" if self.batch_normalize == 1 else "None",
                "batch_norm_params" if self.batch_normalize == 1 else "None", self.act,self.feature_map_num,self.ptr,self.feature_map_num)
        elif self.stride>1:
            return "\t\t{:}=_fixed_padding({:}, {:})\n\t\tx{:} = slim.conv2d({:}, {:}, {:}, stride={:}, normalizer_fn={:}, normalizer_params={:}, activation_fn={:},padding=\"VALID\")".format(self.inputs,self.inputs,self.size,
                self.ptr, self.inputs, self.filters, self.size, self.stride,
                "slim.batch_norm" if self.batch_normalize == 1 else "None",
                "batch_norm_params" if self.batch_normalize == 1 else "None", self.act)
        else:
            return "\t\tx{:} = slim.conv2d({:}, {:}, {:}, stride={:}, normalizer_fn={:}, normalizer_params={:}, activation_fn={:},padding=\"SAME\")".format(
            self.ptr, self.inputs, self.filters, self.size, self.stride,
            "slim.batch_norm" if self.batch_normalize == 1 else "None",
            "batch_norm_params" if self.batch_normalize == 1 else "None", self.act)


class Route:
    def __init__(self, ptr, **kwargs):
        self.groups = int(kwargs.get("groups", -1))
        self.group_id = int(kwargs.get("group_id", -1))
        self.layers = kwargs.get("layers", [])
        if self.groups > 0:
            self.group = True
            self.input = self.layers[0]
        else:
            self.group = False
            self.inputs = self.layers
        self.ptr = ptr

    def __str__(self):
        if self.group:
            return "\t\tx{:} = tf.split({:}, num_or_size_splits={:}, axis=-1)[{:}]".format(self.ptr, self.input, self.groups,
                                                                                       self.group_id)
        elif len(self.inputs)==1:
            return "\t\tx{:} = {:}".format(self.ptr, ", ".join(self.inputs))
        else:
            return "\t\tx{:} = tf.concat([{:}], axis=-1)".format(self.ptr, ", ".join(self.inputs))

class Maxpool:
    def __init__(self,ptr,**kwargs):
        self.size=kwargs["size"]
        self.stride = kwargs["stride"]
        self.ptr=ptr
    def set_input(self, inputs):
        self.inputs = inputs
    def __str__(self):
        return "\t\tx{:} = slim.max_pool2d({:}, kernel_size={:}, stride={:}, padding=\"SAME\")".format(self.ptr,self.inputs,self.size,self.stride)

class Upsample:
    def __init__(self,ptr,stride):
        self.ptr=ptr
        self.stride=stride
    def set_input(self, inputs):
        self.inputs = inputs
    def __str__(self):
        return "\t\tx{} = tf.image.resize_nearest_neighbor({}, (tf.shape({})[1]*{}, tf.shape({})[2]*{}))".format(self.ptr,self.inputs,self.inputs,self.stride,self.inputs,self.stride)

class Shortcut:
    def __init__(self,ptr):
        self.ptr=ptr
    def set_input(self, *inputs):
        self.inputs = inputs
    def __str__(self):
        return "\t\tx{}={}+{}".format(self.ptr,self.inputs[0],self.inputs[1])

if __name__ == "__main__":
    layer = Layer(80)
    variable_list = ["inputs"]
    op_list=[]
    ptr = 0
    yolo_num=0
    print("with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):\n\twith slim.arg_scope([slim.conv2d],biases_initializer=None,weights_regularizer=slim.l2_regularizer(self.weight_decay)):")
    with open("./test.cfg") as cfg:
        line = cfg.readline()
        while line:
            line = line.strip("\n")
            if line == "[convolutional]":
                conv_dict = dict()
                while True:
                    line_tmp=cfg.readline()
                    while line_tmp[0]=="#":
                        line_tmp = cfg.readline()
                    attr = line_tmp.strip("\n").split("=")
                    try:
                        conv_dict[attr[0]] = attr[1]
                    except:
                        break
                conv_layer = ConvLayer(layer, ptr, **conv_dict)
                conv_layer.set_input(variable_list[-1])
                variable_list.append("x{}".format(ptr))
                ptr += 1
                op_list.append(conv_layer)
            if line == "[route]":
                route_dict = dict()
                line_tmp = cfg.readline()
                while line_tmp[0] == "#":
                    line_tmp = cfg.readline()
                layers = [variable_list[int(i)] for i in line_tmp.strip("\n").split("=")[-1].split(",")]
                route_dict["layers"] = layers
                line_tmp = cfg.readline()
                while line_tmp[0] == "#":
                    line_tmp = cfg.readline()
                groups = line_tmp.strip("\n").split("=")[-1]
                if groups:
                    line_tmp = cfg.readline()
                    while line_tmp[0] == "#":
                        line_tmp = cfg.readline()
                    group_id = line_tmp.strip("\n").split("=")[-1]
                    route_dict["groups"] = groups
                    route_dict["group_id"] = group_id
                route = Route(ptr, **route_dict)
                variable_list.append("x{}".format(ptr))
                ptr += 1
                op_list.append(route)
            if line=="[maxpool]":
                maxpool_dict = dict()
                while True:
                    attr = cfg.readline().strip("\n").split("=")
                    try:
                        maxpool_dict[attr[0]] = attr[1]
                    except:
                        break
                maxpool = Maxpool(ptr, **maxpool_dict)
                maxpool.set_input(variable_list[-1])
                variable_list.append("x{}".format(ptr))
                ptr += 1
                op_list.append(maxpool)
            if line=="[upsample]":
                line_tmp = cfg.readline()
                while line_tmp[0] == "#":
                    line_tmp = cfg.readline()
                stride = int(line_tmp.strip("\n").split("=")[-1])
                upsample = Upsample(ptr, stride)
                upsample.set_input(variable_list[-1])
                variable_list.append("x{}".format(ptr))
                ptr += 1
                op_list.append(upsample)
            if line=="[yolo]":
                yolo_num+=1
                variable_list.append("yolo")
                op_list[-1].before_yolo(yolo_num)
            if line=="[shortcut]":
                line_tmp = cfg.readline()
                while line_tmp[0] == "#":
                    line_tmp = cfg.readline()
                _from = int(line_tmp.strip("\n").split("=")[-1])
                shortcut=Shortcut(ptr)
                shortcut.set_input(variable_list[_from],variable_list[-1])
                variable_list.append("x{}".format(ptr))
                ptr += 1
                op_list.append(shortcut)
            line = cfg.readline()
    for op in op_list:
        print(op)
    print("\t\treturn {}".format(", ".join(["feature_map_{}".format(i) for i in range(1,yolo_num+1)])))
