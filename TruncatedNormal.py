from edward.models import RandomVariable
from tensorflow.contrib.distributions import Distribution
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.random_ops import parameterized_truncated_normal
from tensorflow.python.ops.distributions import special_math
import tensorflow as tf
import math


class distributions_TruncatedNormal(Distribution):

    def __init__(self,
                 loc=0.0,
                 scale=1.1,
                 minval=-2.,
                 maxval=2.,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="TruncatedNormal"):
        """Construct Truncated Normal distributions with mean `loc`, stddev `scale`,
            within the interval (`lower_bound`, `upper_bound`).

             Args:
              loc: Floating point tensor; the means of the distribution(s).
              scale: Floating point tensor; the stddevs of the distribution(s).
                Must contain only positive values.
              minval: Floating point tensor; the lower bounds of the range
                for the distribution(s).
              maxval: Floating point tensor; the upper bounds of the range
                for the distribution(s).
              validate_args: Python `bool`, default `False`. When `True`
                distribution parameters are checked for validity despite
                possibly degrading runtime performance. When `False` invalid
                inputs may silently render incorrect outputs.
              allow_nan_stats: Python `bool`, default `True`. When `True`,
                statistics (e.g., mean, mode, variance) use the value "`NaN`"
                to indicate the result is undefined. When `False`, an exception
                is raised if one or more of the statistic's batch members are
                undefined.
              name: Python `str` name prefixed to Ops created by this class.
        """
        parameters = locals()
        with tf.name_scope(name, values=[loc,
                                         scale,
                                         minval,
                                         maxval]):
            with tf.control_dependencies([tf.assert_positive(scale)] if
                                         validate_args else []):
                self._loc = loc
                self._scale = scale
                self._minval = minval
                self._maxval = maxval
                tf.assert_same_float_dtype([self._loc,
                                            self._scale,
                                            self._minval,
                                            self._maxval])
        super(distributions_TruncatedNormal, self).__init__(
            dtype=self._scale.dtype,
            reparameterization_type=tf.contrib.distributions.NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            graph_parents=[self._loc, self._scale, self._minval, self._maxval],
            name=name)

    @staticmethod
    def _param_shapes(sample_shape):
        return dict(
            zip(("loc", "scale", "minval", "maxval"), ([tf.convert_to_tensor(
                sample_shape, dtype=dtypes.int32)] * 4)))

    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        return self._scale

    @property
    def minval(self):
        return self._minval

    @property
    def maxval(self):
        return self._maxval

    def _batch_shape_tensor(self):
        return array_ops.broadcast_dynamic_shape(
            array_ops.shape(self.loc),
            array_ops.broadcast_dynamic_shape(
                array_ops.shape(self.scale),
                array_ops.broadcast_dynamic_shape(
                    array_ops.shape(self.minval),
                    array_ops.shape(self.maxval))))

    def _batch_shape(self):
        return array_ops.broadcast_static_shape(
            self.loc.get_shape(),
            array_ops.broadcast_static_shape(
                self.scale.get_shape(),
                array_ops.broadcast_static_shape(
                    self.minval.get_shape(),
                    self.maxval.get_shape())))

    def _event_shape_tensor(self):
        return tf.constant_op.constant([], dtype=math_ops.int32)

    def _event_shape(self):
        return tensor_shape.scalar()

    def _sample_n(self, n, seed=None):
        shape = array_ops.concat([[n], self.batch_shape_tensor()], 0)
        samples = parameterized_truncated_normal(shape,
                                                 means=self.loc,
                                                 stddevs=self.scale,
                                                 minvals=self.minval,
                                                 maxvals=self.maxval,
                                                 dtype=self.loc.dtype,
                                                 seed=seed)
        return samples

    def _log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _log_unnormalized_prob(self, x):
        return -0.5 * math_ops.square(self._z(x))

    def _log_normalization(self):
        normal_const = 0.5 * math.log(2. * math.pi) + \
            math_ops.log(self.scale)
        trunc_const = math_ops.log(self._normal_cdf(self._z(self.maxval)) -
                                   self._normal_cdf(self._z(self.minval)))
        return normal_const + trunc_const

    def _normal_cdf(self, x):
        return special_math.ndtr(self._z(x))

    def _z(self, x):
        """Standardize input `x` to a unit normal."""
        with tf.name_scope("standardize", values=[x]):
            return (x - self.loc) / self.scale

    def _inv_z(self, z):
        """Reconstruct input `x` from its normalized version."""
        with tf.name_scope("reconstruct", values=[z]):
            return z * self.scale + self.loc

class TruncatedNormal(RandomVariable, distributions_TruncatedNormal):
    def __init__(self, *args, **kwargs):
        #super(CustomRandomVariable,self).__init__(loc, scale, sample_shape)
        #distributions_CustomRandomVariable.__init__(self, loc, scale, sample_shape)
        RandomVariable.__init__(self, *args, **kwargs)
