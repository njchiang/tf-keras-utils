"""
Some metrics for comparing two waveforms (e.g., original waveform against reconstructed)
"""
import tensorflow as tf

# import numpy as np
# from scipy.stats import pearsonr

# def batch_pearson(y_true, y_pred):
#     assert y_true.shape == y_pred.shape, "Predicted and labels have different shapes"
#     return np.mean([pearsonr(yt.squeeze(), yp.squeeze())[0] for yt, yp in zip(y_true, y_pred)])

class PearsonCorrelation(tf.keras.metrics.Metric):
    """ 
    streaming pearson correlation
    sum(x * y) - n * mx * my
    -----------------------
    """
    def __init__(self,  name="pearson_correlation", eps=1e-7, **kwargs):
        super(PearsonCorrelation, self).__init__(name=name, **kwargs)
        self._sample_count = self.add_weight("n_samples", initializer="zeros")
        self._true_sum = self.add_weight("sum_labels", initializer="zeros")
        self._pred_sum = self.add_weight("sum_preds", initializer="zeros")
        self._prod_sum = self.add_weight("sum_products", initializer="zeros")
        self._true_ssq = self.add_weight("ssq_labels", initializer="zeros")
        self._pred_ssq = self.add_weight("ssq_preds", initializer="zeros")
        self._eps = eps
        self.pearson_correlation = self.add_weight(name="pearson", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true = tf.reshape(y_true, (-1, 1))
        # y_pred = tf.reshape(y_pred, (-1, 1))
        self._sample_count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
        self._true_sum.assign_add(tf.reduce_sum(y_true))
        self._pred_sum.assign_add(tf.reduce_sum(y_pred))
        self._prod_sum.assign_add(tf.reduce_sum(y_true * y_pred))
        self._true_ssq.assign_add(tf.reduce_sum(tf.pow(y_true, 2)))
        self._pred_ssq.assign_add(tf.reduce_sum(tf.pow(y_pred, 2)))
        # mx = self._true_sum / self._sample_count # tf.math.reduce_mean(y_true)
        # sx = tf.math.reduce_std(y_true)
        # my = self._pred_sum / self._sample_count # tf.math.reduce_mean(y_pred)
        # sy = tf.math.reduce_std(y_pred)
        numerator = (self._prod_sum * self._sample_count) - (self._true_sum * self._pred_sum)  
        sx = tf.sqrt((self._sample_count * self._true_ssq) - tf.pow(self._true_sum, 2))
        sy = tf.sqrt((self._sample_count * self._pred_ssq) - tf.pow(self._pred_sum, 2)) 
        denominator = (sx * sy) + self._eps

        self.pearson_correlation.assign(numerator / denominator)
    
    def result(self):
        return self.pearson_correlation

    def reset_states(self):
        self.pearson_correlation.assign(0.)
        self._sample_count.assign(0.)
        self._true_sum.assign(0.)
        self._pred_sum.assign(0.)
        self._prod_sum.assign(0.)
        self._true_ssq.assign(0.)
        self._pred_ssq.assign(0.)


class PercentRMSDifference(tf.keras.metrics.Metric):
    """ 
    Used in Yildirim et al., 2018 
    sqrt(sum((y_true - y_pred)**2) / sum(y_true ** 2))
    """
    def __init__(self, name="percent_rms_diff", eps=1e-7, **kwargs):
        super(PercentRMSDifference, self).__init__(name=name, **kwargs)
        self._ssq_diff = self.add_weight(name="ssq_diff", initializer="zeros")
        self._ssq_val = self.add_weight(name="ssq_val", initializer="zeros")
        self.prd = self.add_weight(name="prd", initializer="zeros")
        self._eps = eps

    def update_state(self, y_true, y_pred, sample_weight=None):
        self._ssq_diff.assign_add(tf.reduce_sum(tf.pow(y_true - y_pred, 2)))
        self._ssq_val.assign_add(tf.reduce_sum(tf.pow(y_true, 2)))
        prd = 100. * tf.sqrt(self._ssq_diff / (self._ssq_val + self._eps))
        self.prd.assign(prd)

    def result(self):
        return self.prd

    def reset_states(self):
        self.prd.assign(0.)
        self._ssq_diff.assign(0.)
        self._ssq_val.assign(0.)


class PRDNormalized(PercentRMSDifference):
    """ 
    Used in Yildirim et al., 2018 
    sqrt(sum((y_true - y_pred)**2) / sum((y_true - mean(y_true)) ** 2))
    """
    def __init__(self, name="prd_normalized", **kwargs):
        super(PRDNormalized, self).__init__(name=name, **kwargs)
        self.prdn = self.add_weight(name="prdn", initializer="zeros")
        self._sample_count = self.add_weight("n_samples", initializer="zeros")
        self._true_sum = self.add_weight("true_sum", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self._sample_count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
        self._true_sum.assign_add(tf.reduce_sum(y_true))

        self._ssq_diff.assign_add(tf.reduce_sum(tf.pow(y_true - y_pred, 2)))

        # TODO : This is wrong
        # self._ssq_val.assign_add(tf.reduce_sum(tf.pow(y_true, 2) - self._sample_count * tf.pow(self._true_sum, 2)))
        # E(X^2) - E(X)^2
        self._ssq_val.assign_add(tf.reduce_sum(tf.pow(y_true, 2) - tf.pow(self._true_sum, 2)))


        prdn = 100. * tf.sqrt(self._ssq_diff / (self._ssq_val + self._eps))
        self.prdn.assign(prdn)

    def result(self):
        return self.prdn

    def reset_states(self):
        super().reset_states()
        self.prdn.assign(0.)
        self._sample_count.assign(0.)
        self._true_sum.assign(0.)


class SignalNoiseRatio(PRDNormalized):
    """
    Used in Yildirim et al., 2018
    signal to noise ratio
    """
    def __init__(self, name="snr", **kwargs):
        super(SignalNoiseRatio, self).__init__(name=name, **kwargs)
        self.snr = self.add_weight(name="signal_to_noise", initializer="zeros")
       
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true, y_pred, sample_weight)
        snr = 10. * tf.math.log(self._ssq_val / (self._ssq_diff + self._eps))
        self.snr.assign(snr)
    
    def result(self):
        return self.snr
    
    def reset_states(self):
        super().reset_states()
        self.snr.assign(0.)