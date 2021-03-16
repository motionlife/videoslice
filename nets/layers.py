"""
Author : Hao Xiong
Tensorflow Implementation of Bilinear Additive Upsampling.
Reference : https://arxiv.org/abs/1707.05847
"""
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K


def BilinearAdditiveUpsampling(x, to_channel_num):
    from_channel_num = x.get_shape().as_list()[3]
    assert from_channel_num % to_channel_num == 0
    channel_split = from_channel_num // to_channel_num

    new_shape = x.get_shape().as_list()
    new_shape[1] *= 2
    new_shape[2] *= 2
    new_shape[3] = to_channel_num

    upsampled_x = tf.image.resize(x, new_shape[1:3])

    output_list = []
    for i in range(to_channel_num):
        splited_upsampled_x = upsampled_x[:, :, :, i * channel_split:(i + 1) * channel_split]
        output_list.append(tf.reduce_sum(splited_upsampled_x, axis=-1))

    output = tf.stack(output_list, axis=-1)

    return output


class AdditiveUpSampling2D(Layer):
    def __init__(self, out_channel, size=(2, 2), interpolation='bilinear', data_format=None, **kwargs):
        super(AdditiveUpSampling2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.out_channel = out_channel
        if interpolation not in {'nearest', 'bilinear'}:
            raise ValueError('`interpolation` argument should be one of `"nearest"` or `"bilinear"`.')
        self.interpolation = interpolation
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return tensor_shape.TensorShape([input_shape[0], self.out_channel, height, width])
        else:
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return tensor_shape.TensorShape([input_shape[0], height, width, self.out_channel])

    def call(self, inputs):
        x = K.resize_images(inputs, self.size[0], self.size[1], self.data_format, interpolation=self.interpolation)
        b, h, w, c = K.int_shape(x)
        x = array_ops.reshape(x, [-1, h, w, self.out_channel, tf.math.floordiv(c, self.out_channel)])
        return math_ops.reduce_sum(x, -1)

    def get_config(self):
        config = {
            'size': self.size,
            'out_channel': self.out_channel,
            'data_format': self.data_format,
            'interpolation': self.interpolation
        }
        base_config = super(AdditiveUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    img = tf.random.normal([20, 100, 100, 20])
    y = BilinearAdditiveUpsampling(img, to_channel_num=5)
    z = AdditiveUpSampling2D(out_channel=5, size=(2, 2))(img)
    tf.assert_equal(y, z)
