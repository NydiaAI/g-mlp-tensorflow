from tensorflow.keras.layers import Layer, LayerNormalization
import tensorflow as tf

class PreNorm(Layer):
    def __init__(self, fn, **kwargs):
        self.fn = fn
        return super(PreNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        self.norm = LayerNormalization(input_shape=input_shape)

    def call(self, x):
        self.fn(self.norm(x))