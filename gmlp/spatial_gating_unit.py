import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import Layer, LayerNormalization

class SpatialGatingUnit(Layer):
    def __init__(self, 
                dim_seq, 
                causal = False, 
                activation = None, 
                init_eps = 1e-3,
                kernel_regularizer=None,
                bias_regularizer=None):
        
        self.dim_seq = dim_seq
        self.causal = causal
        self.activation = activation
        self.init_eps = init_eps / dim_seq

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        return super(SpatialGatingUnit, self).__init__()

    def build(self, _):

        self.conv1d_bias = self.add_weight(
            name="sgu_conv1d_bias", 
            regularizer=self.bias_regularizer,
            shape=[self.dim_seq], 
            initializer=tf.ones
        )
        
        self.conv1d_kernel = self.add_weight(
            name="sgu_conv1d_kernel", 
            regularizer=self.kernel_regularizer,
            shape=(1, self.dim_seq, self.dim_seq), 
            initializer=RandomUniform(minval=-self.init_eps, maxval=self.init_eps)
        )

        self.norm = LayerNormalization()

    def call(self, x):
        n = x.shape[1]
        weight, bias = self.conv1d_kernel, self.conv1d_bias
        if(self.causal):
            weight, bias = weight[:, :n, :n], bias[:n]

            mask = tf.ones(weight.shape[1:])

            # band_part and set_diag replace triu(1) in lucidrains' implementation
            mask = tf.linalg.band_part(mask, 0, -1)
            mask_diag = tf.linalg.diag_part(mask)
            mask = tf.linalg.set_diag(mask, tf.zeros_like(mask_diag))
            
            mask = tf.cast(mask, dtype=tf.bool)
            weight = tf.where(mask[None, ...], tf.zeros_like(weight), weight)


        res, gate = tf.split(x, 2, axis=-1)
        gate = self.norm(gate)
        data_format = "NWC"
        conv1d_kwargs = {
            "stride": 1, 
            "use_cudnn_on_gpu": True, 
            "data_format": data_format,
            "padding": "VALID"
        }

        gate = tf.transpose(gate, (0,2,1))
        gate = tf.nn.conv1d(gate, filters=self.conv1d_kernel, **conv1d_kwargs) 
        gate = tf.nn.bias_add(gate, self.conv1d_bias, data_format=data_format) # Now add bias
        gate = tf.transpose(gate, (0,2,1))
        
        if(self.activation is not None):
            gate = self.activation(gate)
        return gate * res
        
