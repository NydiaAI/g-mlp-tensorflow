import tensorflow as tf
from tensorflow.keras import Model
from gmlp.gmlp import gMLP
from gmlp.sequential import SequentialLayer
from tensorflow.keras.layers import Embedding, LayerNormalization, Dense

class NLPgMLPModel(Model):
    def __init__(self, 
        depth,
        embedding_dim,
        num_tokens,
        seq_length,
        **kwargs):
        super(NLPgMLPModel, self).__init__()

        self.to_embed = Embedding(num_tokens, embedding_dim, input_length=seq_length)

        self.gmlp = gMLP(
            depth=depth, 
            seq_length=seq_length,
            activation=tf.nn.swish,
            **kwargs)

        self.to_logits = SequentialLayer([
            LayerNormalization(),
            Dense(1, activation="linear")
        ])

    def call(self, input, training=False):
        x = self.to_embed(input)
        x = self.gmlp(x)
        x = self.to_logits(x)
        return tf.math.sign(x)
