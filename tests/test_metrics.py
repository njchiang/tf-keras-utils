import pytest
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from tfutils import losses
from tfutils import metrics


yt = np.random.rand(1, 100).astype(np.float32)
yp = np.random.rand(1, 100).astype(np.float32)
ytrs = np.reshape(yt, (1, -1))
yprs = np.reshape(yp, (1, -1))
yt_tf = tf.convert_to_tensor(yt)
yp_tf = tf.convert_to_tensor(yp)

def test_pearson():
    tf_result = metrics.PearsonCorrelation(eps=0)(yp_tf, yt_tf).numpy()
    np_result = np.corrcoef(ytrs, yprs)[0, 1]
    assert np.allclose(np_result, tf_result), "numpy ({:.2f}) and tf ({:.2f}) don't line up".format(np_result, tf_result)

def test_percentrmsdiff():
    np_result = 100. * np.sqrt( np.sum((ytrs - yprs)**2) / np.sum(ytrs ** 2) )
    tf_result = metrics.PercentRMSDifference(eps=0)(yt_tf, yp_tf).numpy()
    assert np.allclose(np_result, tf_result), "numpy ({:.2f}) and tf ({:.2f}) don't line up".format(np_result, tf_result)

def test_prdnorm():
    np_result = 100. * np.sqrt( np.sum((ytrs - yprs)**2) / np.sum((ytrs - ytrs.mean()) ** 2) )
    tf_result = metrics.PRDNormalized(eps=0)(yt_tf, yp_tf).numpy()
    assert np.allclose(np_result, tf_result), "numpy ({:.2f}) and tf ({:.2f}) don't line up".format(np_result, tf_result)

def test_snr():
    np_result = 10. * np.log( np.sum((ytrs - ytrs.mean()) ** 2) / np.sum((ytrs - yprs)**2) )
    tf_result = metrics.SignalNoiseRatio(eps=0)(yt_tf, yp_tf).numpy()
    assert np.allclose(np_result, tf_result), "numpy ({:.2f}) and tf ({:.2f}) don't line up".format(np_result, tf_result)
    
    # # test metrics
    # print(metrics.PearsonCorrelation()(pred, y))

    # print(metrics.PercentRMSDifference()(pred, y))

    # print(metrics.PRDNormalized()(pred, y))

    # print(metrics.SignalNoiseRatio()(pred, y))
    # # test 

