from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential, Model
from gmlp.activations.gelu import GELU
from gmlp.spatial_gating_unit import SpatialGatingUnit

class gMLPBlock(Model):
    def __init__(self, 
                dim,
                dim_ff,
                seq_len,
                causal=False,
                activation=None,
                kernel_regularizer=None,
                bias_regularizer=None,
                **kwargs):
        super(gMLPBlock, self).__init__(**kwargs)

        reg_kwargs = {
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer
        }

        self.proj_in = Sequential([
            Dense(dim_ff, activation="linear", **reg_kwargs),
            GELU()
        ])

        self.sgu = SpatialGatingUnit(
            seq_len, 
            causal=causal, 
            activation=activation,
            **reg_kwargs
        )

        self.proj_out = Dense(dim, activation="linear", **reg_kwargs)
        
    def call(self, x, training=False):
        x = self.proj_in(x)
        x = self.sgu(x)
        x = self.proj_out(x)
        return x