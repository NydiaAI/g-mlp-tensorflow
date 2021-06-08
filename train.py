from gmlp.model.nlp_gmlp import NLPgMLPModel
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow.keras.datasets.imdb as imdb

train_set, val_set = imdb.load_data()

model = NLPgMLPModel(
    depth=5, 
    embedding_dim=256, 
    num_tokens=88584, 
    seq_length=2500,
    ff_mult=4)

model.compile(optimizer='adam', loss=BinaryCrossentropy(from_logits=False))

