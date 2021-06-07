from gmlp.pre_norm import PreNorm
import tensorflow as tf

from tensorflow.keras.layers import Layer

from gmlp.sequential import SequentialLayer
from gmlp.residual import Residual
from gmlp.gmlp_block import gMLPBlock

class gMLP(Layer): 
    def __init__(self, 
                depth,
                seq_length,
                ff_mult=4, 
                training=False,
                causal=False,
                activation=None,
                dropout_ratio=0.2,
                **kwargs):
        
        self.dropout_ratio=dropout_ratio
        self.training = training
        self.activation = activation
        self.ff_mult = ff_mult
        self.seq_length = seq_length
        self.depth = depth
        self.causal = causal

        return super(gMLP, self).__init__(**kwargs)
    
    def build(self, input_shape):
        dim = input_shape[-1]

        dim_ff = dim * self.ff_mult

        self.layers = SequentialLayer([
            Residual(
                PreNorm(
                    gMLPBlock(
                        dim_ff=dim_ff, 
                        seq_len=self.seq_len, 
                        causal=self.causal, 
                        activation= self.activation
                    )
                )
            ) for _ in range(self.depth) ])
        
    def call(self, x):
        return self.layers(x)
