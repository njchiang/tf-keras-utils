import tensorflow as tf

from tfutils import losses
from tfutils import metrics


def main():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)
    ])
    x = tf.random.normal(shape=(2, 10))
    y = tf.random.normal(shape=(2,))

    # test ae_loss_fn
    _, pred = losses.ae_loss_fn(model, x, y)

    # test metrics
    print(metrics.PearsonCorrelation()(pred, y))

    print(metrics.PercentRMSDifference()(pred, y))

    print(metrics.PRDNormalized()(pred, y))

    print(metrics.SignalNoiseRatio()(pred, y))
    # test 


if __name__ == "__main__":
    tf.enable_eager_execution()
    main()