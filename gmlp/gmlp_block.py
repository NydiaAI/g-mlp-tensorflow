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
                **kwargs):
        super(gMLPBlock, self).__init__(**kwargs)    

        self.proj_in = Sequential([
            Dense(dim_ff, activation="linear"),
            GELU()
        ])

        self.sgu = SpatialGatingUnit(
            seq_len, 
            causal=causal, 
            activation=activation
        )

        self.proj_out = Dense(dim, activation="linear")
        
    def call(self, x, training=False):
        x = self.proj_in(x)
        x = self.sgu(x)
        x = self.proj_out(x)
        return x