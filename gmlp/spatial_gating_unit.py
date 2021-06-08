import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization

class SpatialGatingUnit(Layer):
    def __init__(self, 
                dim_seq, 
                causal = False, 
                activation = None, 
                init_eps = 1e-3,
                **kwargs):
        
        self.dim_seq = dim_seq
        self.causal = causal
        self.activation = activation
        self.init_eps = init_eps / dim_seq

        return super(SpatialGatingUnit, self).__init__(**kwargs)

    def build(self, _):
        self.conv1d_bias = tf.Variable(
            tf.ones(shape=[self.dim_seq]), 
            name="sgu_conv1d_bias"
        )

        # Finally was able to confirm the shape that the filter must have in tf.nn.conv1d (reversed order from pytorch)
        # https://stackoverflow.com/questions/38114534/basic-1d-convolution-in-tensorflow
        self.conv1d_kernel = tf.Variable(
            tf.random.uniform(
                shape=(1, self.dim_seq, self.dim_seq),
                minval=-self.init_eps, 
                maxval=self.init_eps
            ), 
            name="sgu_conv1d_kernel"
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
            weight = tf.where(mask[None, ...], 0., weight)


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
        
