from tensorflow.keras.layers import Layer, Dense
from gmlp.activations.gelu import GELU
from gmlp.sequential import SequentialLayer
from gmlp.spatial_gating_unit import SpatialGatingUnit

class gMLPBlock(Layer):
    def __init__(self, 
                dim_ff,
                seq_len,
                causal=False,
                activation=None,
                **kwargs):
        
        self.activation = activation
        self.dim_ff = dim_ff
        self.seq_len = seq_len
        self.causal = causal

        return super(gMLPBlock, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.layers = SequentialLayer([
            SequentialLayer([
                Dense(self.dim_ff, input_shape=input_shape, activation="linear"),
                GELU()
            ]),
            SpatialGatingUnit(
                self.seq_len, 
                causal=self.causal, 
                activation=self.activation
            ),
            Dense(input_shape[-1], activation="linear")
        ])

    def call(self, x):
        return self.layers(x)