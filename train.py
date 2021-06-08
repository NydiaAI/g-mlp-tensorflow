from gmlp.model.nlp_gmlp import NLPgMLPModel
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.data import Dataset

import tensorflow as tf
import numpy as np
import tensorflow.keras.datasets.imdb as imdb

train_set, val_set = imdb.load_data()

model = NLPgMLPModel(
    depth=5, 
    embedding_dim=256, 
    num_tokens=88584, 
    seq_length=2500,
    ff_mult=4)

model.compile(optimizer='adam', loss=BinaryCrossentropy(from_logits=False))
def gen(set):
    def iter():
        values, labels = set
        for i in range(len(values)):
            review = np.array(values[i], dtype="int32")
            label = np.array(labels[i], dtype="int32")

            yield (review, label)

    return iter

ds_args = ((tf.int64, tf.int64), (tf.TensorShape([None]), tf.TensorShape([])))
train_ds = Dataset.from_generator(gen(train_set), *ds_args)
val_ds = Dataset.from_generator(gen(val_set), *ds_args)

model.fit(x=train_ds, validation_data=val_ds, epochs=10)