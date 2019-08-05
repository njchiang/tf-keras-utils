import os
import tensorflow as tf
from absl import logging


def keras_train(ds, model, loss_fn, opt, epochs, steps_per_epoch=None, log_every=100, save_every=-1, checkpoint_dir="/tmp", initial_epoch=0, metrics=None):
    checkpoint_path = os.path.join(checkpoint_dir, "cp-{epoch:04d}.ckpt")
    log_dir = os.path.join(checkpoint_dir, "logs")
    # This is weird because epoch is not recovered...
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            load_weights_on_restart=True),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_grads=True,
            update_freq=log_every)
    ]
    if save_every:
        tmp_checkpoint_path = os.path.join(checkpoint_dir, "cp-tmp.ckpt")
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=tmp_checkpoint_path,
        save_freq=save_every))
    
    keras_metrics = list(metrics.values())

    model.compile(opt, loss=loss_fn, metrics=keras_metrics)
    for e in range(initial_epoch, initial_epoch + epochs):
        # TODO : one patient per epoch?
        _ = model.fit(ds.dataset, epochs=e+1, callbacks=callbacks, initial_epoch=e, verbose=2, steps_per_epoch=steps_per_epoch)
        with open(os.path.join(checkpoint_dir, "EPOCH"), "w") as f:
            f.write(str(e+1))


def train_step(model, x, y, loss_fn, optimizer):
    # doesn't work in eager mode with tf 1.14
    with tf.GradientTape() as tape:
        # loss function takes in model, x, y and can compute more complex losses
        loss, pred = loss_fn(model, x, y, training=True)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, pred


# def keras_train(ds, model, loss_fn, opt, epochs, steps_per_epoch=None, log_every=100, save_every=-1, checkpoint_dir="/tmp", initial_epoch=0, metrics=None):
# for now, only goes through entire dataset
def train_loop(ds, model, loss_fn, opt, epochs, steps_per_epoch=None, log_every=100, save_every=-1, checkpoint_dir="/tmp", initial_epoch=0, metrics=None):
   for e in range(initial_epoch, initial_epoch + epochs):
        for d in ds.dataset:
            x, y = d
            loss, pred = train_step(model, x, y, loss_fn, opt)

            if opt.iterations.numpy() % log_every == 0:
                metrics_list = ["{}: {:.3f}".format(n.name, f(y, pred)) for n, f in metrics.items()]
                metrics_str = " | ".join(metrics_list)
                logging.info("Epoch {} | Step {} | Loss: {} | {}".format(e, opt.iterations.numpy(), loss, metrics_str))
            
            if save_every:
                if opt.iterations.numpy() % save_every == 0:
                    logging.info("Saving...")
                    checkpoint_path = os.path.join(checkpoint_dir, "cp-tmp.ckpt")
                    model.save_weights(checkpoint_path)

        checkpoint_path = os.path.join(checkpoint_dir, "cp-{:04d}.ckpt".format(e+1))
        logging.info("Epoch {}: Saving to {}".format(e, checkpoint_path))
        model.save_weights(checkpoint_path)
        with open(os.path.join(checkpoint_dir, "EPOCH"), "w") as f:
            f.write(str(e+1))


