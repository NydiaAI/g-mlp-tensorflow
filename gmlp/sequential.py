import tensorflow as tf
from tensorflow.keras.layers import Layer

class SequentialLayer(Layer):
    def __init__(self, layers, **kwargs):
        self.layers = layers
        return super(SequentialLayer, self).__init__(**kwargs)
    
    def call(self, x):
        for layer in self.layers: 
            x = layer(x)
        return x