import tensorflow as tf
from tensorflow.keras import Model, Sequential
from gmlp.gmlp import gMLP
from tensorflow.keras.layers import Embedding, LayerNormalization, Dense, Flatten

class NLPgMLPModel(Model):
    def __init__(self, 
        depth,
        embedding_dim,
        num_tokens,
        seq_len,
        **kwargs):
        super(NLPgMLPModel, self).__init__()

        self.to_embed = Embedding(num_tokens, embedding_dim, input_length=seq_len)

        self.gmlp = gMLP(
            dim=embedding_dim,
            depth=depth, 
            seq_len=seq_len,
            activation=tf.nn.swish,
            **kwargs)

        self.to_logits = Sequential([
            Flatten(data_format="channels_first"),
            LayerNormalization(),
            Dense(1, activation="sigmoid")
        ])

    def call(self, input, training=False):
        x = tf.cast(input, dtype="int64")
        x = self.to_embed(input)
        x = self.gmlp(x)
        x = self.to_logits(x)
        return x
