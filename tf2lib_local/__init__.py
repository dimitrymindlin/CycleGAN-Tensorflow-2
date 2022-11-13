import tensorflow as tf

from tf2lib_local.data import *
from tf2lib_local.image import *
from tf2lib_local.ops import *
from tf2lib_local.utils import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)
