from gmlp.model.nlp_gmlp import NLPgMLPModel
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.data import Dataset

from tensorflow.train import AdamOptimizer

import tensorflow as tf
import numpy as np
import tensorflow.keras.datasets.imdb as imdb

SEQ_LEN = 768
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-3

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

train_data, val_data = imdb.load_data()

model = NLPgMLPModel(
    depth=5, 
    embedding_dim=512, 
    num_tokens=88585, 
    seq_len=SEQ_LEN,
    causal=True,
    ff_mult=4)

model.compile(optimizer=AdamOptimizer(learning_rate=LEARNING_RATE), loss=BinaryCrossentropy(from_logits=False))

def gen(set):
    def iter():
        values, labels = set
        for i in range(len(values)):
            review = np.array(values[i], dtype="int32")
            label = np.array(labels[i], dtype="int32")

            review_length = review.shape[0]
            if(review_length < SEQ_LEN):
                pad_length = SEQ_LEN - review_length
                review = np.pad(review, (0, pad_length), constant_values=0.)
            elif(review_length > SEQ_LEN):
                review = review[:SEQ_LEN]

            yield (review, label)

    return iter

def make_dataset(s):
    ds_args = ((tf.int64, tf.int64), (tf.TensorShape([SEQ_LEN]), tf.TensorShape([])))
    ds = Dataset.from_generator(gen(s), *ds_args)
    ds = ds.batch(BATCH_SIZE)
    return ds

train_ds = make_dataset(train_data)
val_ds = make_dataset(val_data)

train_steps = len(train_data[0])
val_steps = len(val_data[0])

model.fit(
    x = train_ds, 
    validation_data = val_ds,
    steps_per_epoch = train_steps // BATCH_SIZE,
    validation_steps = val_steps // BATCH_SIZE,
    epochs=EPOCHS
)