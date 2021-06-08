from tensorflow.keras.layers import Layer, Dense
import tensorflow as tf
from math import pi

class GELU(Layer):
    def __init__(self, **kwargs):
        return super(GELU, self).__init__(**kwargs)
    
    def call(self, x):
        return (x/2.)*(1 + tf.math.tanh(
            tf.math.sqrt(2./tf.constant(pi)) * (x + 0.044715*tf.math.pow(x, 3))
        ))
