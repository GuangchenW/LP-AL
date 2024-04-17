import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib

print(tf.config.list_physical_devices('GPU'))
def get_devices():
    ld = device_lib.list_local_devices()
    return [x.name for x in ld]
print(get_devices())
