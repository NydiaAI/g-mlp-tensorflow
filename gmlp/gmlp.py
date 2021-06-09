from gmlp.pre_norm import PreNorm
import tensorflow as tf

from tensorflow.keras import Sequential, Model

from gmlp.residual import Residual
from gmlp.gmlp_block import gMLPBlock

class gMLP(Model): 
    def __init__(self,
                dim, 
                depth,
                seq_len,
                ff_mult=4, 
                causal=False,
                activation=None,
                dropout_ratio=0.2,
                **kwargs):
        
        super(gMLP, self).__init__(**kwargs)

        dim_ff = dim * ff_mult

        self.residual_layers = Sequential([
            Residual(
                PreNorm(
                    gMLPBlock(
                        dim=dim,
                        dim_ff=dim_ff, 
                        seq_len=seq_len, 
                        causal=causal, 
                        activation= activation
                    )
                )
            ) for _ in range(depth) ])
        
    def call(self, x, training=False):
        return self.residual_layers(x)
