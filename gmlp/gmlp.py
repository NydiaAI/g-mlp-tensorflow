from gmlp.pre_norm import PreNorm
import tensorflow as tf
from tensorflow.keras.layers import Dropout

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

        reg_keys = ['bias_regularizer', 'kernel_regularizer'] 

        for k in reg_keys:
            if(k not in kwargs):
                kwargs[k] = None

        regularization_kwargs = { k: kwargs.pop(k) for k in reg_keys}  
        
        super(gMLP, self).__init__(**kwargs)

        dim_ff = dim * ff_mult


        layers = []
        for _ in range(depth):
            layers.append(
                Residual(
                    PreNorm(
                        gMLPBlock(
                            dim=dim,
                            dim_ff=dim_ff, 
                            seq_len=seq_len, 
                            causal=causal, 
                            activation=activation,
                            **regularization_kwargs
                        )
                    )
                )
            )
            layers.append(Dropout(dropout_ratio))
                

        self.residual_layers = Sequential(layers)
        
    def call(self, x, training=False):
        return self.residual_layers(x, training=training)
