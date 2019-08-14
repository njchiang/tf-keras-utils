import pytest

import numpy as np
import tensorflow as tf

from tfutils.losses import ae_loss_fn, vae_loss_fn


def test_ae_loss_fn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)
    ])
    
    x = tf.random.normal(shape=(2, 10))
    y = tf.random.normal(shape=(2, 1))

    loss, pred = ae_loss_fn(model, x, y)
    # check that predicted output and loss are the correct shape
    assert pred.shape == y.shape, "predicted shape: {}, expected: {}".format(pred.shape, y.shape)