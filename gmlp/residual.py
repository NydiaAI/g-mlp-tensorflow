import tensorflow as tf
from tensorflow.keras.layers import Layer

class Residual(Layer):
    def __init__(self, fn, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn
        
    def call(self, x):
        return self.fn(x) + x