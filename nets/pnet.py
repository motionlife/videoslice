import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Activation, BatchNormalization


def conv2d(filters, kernel, strides, transpose=False, bn=True, relu=True, name='conv2d'):
    def layer_wrapper(x):
        x = (Conv2DTranspose(filters, kernel, strides, padding='same', name=name)
             if transpose else Conv2D(filters, kernel, strides, padding='same', name=name))(x)
        if bn:
            x = BatchNormalization(name=f'{name}_bn')(x)
        if relu:
            x = Activation('relu', name=f'{name}_relu')(x)
        return x

    return layer_wrapper


def conv2dx2(filters, kernel, strides=2, transpose=False, bn=True, relu=True, name='conv2dx2', cat=True):
    def layer_wrapper(x, rem=None):
        x = conv2d(filters, kernel, strides, transpose, bn, relu, name + "_1")(x)
        if rem is not None:
            x = tf.concat([x, rem], -1) if cat else tf.add(x, rem)
        x = conv2d(filters, 3, 1, False, bn, relu, name + "_2")(x)
        return x

    return layer_wrapper


def PairwiseNet(name='pairwise-net'):
    """
    Updated edge-net, hourglass structure:
    1) Increased the depth to enlarge receptive field;
    2) Added batch normalization layer after convolution layers, consistent with unary disparity net.
    """
    ref = Input(shape=(None, None, 3))
    x = conv2d(32, 3, 1, name='basic')(ref)
    x = rem0 = conv2dx2(32, 4, name='conv2d_1')(x)  # 1/2
    x = rem1 = conv2dx2(48, 3, name='conv2d_2')(x)  # 1/4
    x = rem2 = conv2dx2(64, 3, name='conv2d_3')(x)  # 1/8

    x = rem3 = conv2dx2(48, 3, transpose=True, name='conv2dt_4')(x, rem1)  # 1/4
    x = rem4 = conv2dx2(32, 3, transpose=True, name='conv2dt_5')(x, rem0)  # 1/2

    x = rem5 = conv2dx2(48, 3, name='conv2d_6')(x, rem3)  # 1/4
    x = conv2dx2(64, 3, name='conv2d_7')(x, rem2)  # 1/8

    x = conv2dx2(48, 3, transpose=True, name='conv2dt_8')(x, rem5)  # 1/4
    x = conv2dx2(32, 3, transpose=True, name='conv2dt_9')(x, rem4)  # 1/2
    x = conv2dx2(32, 4, transpose=True, name='conv2dt_10')(x)  # original size

    x = tf.abs(conv2d(2, 3, 1, bn=False, relu=False, name='edge-weights')(x))  # [B, H, W, 2]
    x1 = tf.concat([x[:, :-1, :, 0], tf.zeros(shape=[tf.shape(x)[0], 1, tf.shape(x)[2]])], axis=1)
    x2 = tf.concat([x[:, :, :-1, 1], tf.zeros(shape=[tf.shape(x)[0], tf.shape(x)[1], 1])], axis=2)
    out = tf.stack([x1, x2], axis=-1)
    return tf.keras.Model(inputs=ref, outputs=out, name=name)
