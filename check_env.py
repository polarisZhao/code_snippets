# _*_ coding:utf-8 _*_
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
def check_env():
    """
        Check for Tensorflow Version and GPU Support.
    """
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
