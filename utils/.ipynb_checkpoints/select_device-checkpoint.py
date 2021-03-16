import tensorflow as tf
import os


def select_device(n):
    """Select which device(GPU) to use, n = -1 means using CPU only"""
    if n < 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        gpu = tf.config.list_physical_devices('GPU')
        if gpu:
            try:
                tf.config.set_visible_devices(gpu[n], 'GPU')  # only use /gpu:0
            except RuntimeError as e:
                print(e)
