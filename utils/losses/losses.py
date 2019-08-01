import tensorflow as tf

def ae_loss_fn(model, x, y, training=None):
    pred = model(x, training)
    mse = tf.keras.losses.MSE(y, pred)
    return tf.reduce_mean(mse), pred

def vae_loss_fn(model, x, y, training=None):
    z, m, v = model.encoder(x, training)
    pred = model.decoder(z)
    mse = tf.reduce_sum(tf.keras.losses.MSE(y, pred))
    kld = -0.5 * tf.reduce_sum(1 + v - tf.pow(m, 2) - tf.exp(v))
    return mse + kld, pred