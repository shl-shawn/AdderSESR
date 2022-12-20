import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils


@tf.custom_gradient
def adder2d_im2col(X_col, W_col):
    """adder2d_im2col
    Inference function for Adder2D layer.
    # Arguments
        X: tf.Tensor, inputs with default shape (N_X, H, W, C_in).
        W: tf.Tensor, kernels with default shape (h_filter, w_filter, C_in, N_filters)
        stride:
        padding: "valid" or "same".
    # Returns
        outputs (tensor): outputs tensor as input to the next layer
    """

    # adder conv
    outputs = tf.abs((tf.expand_dims(W_col, 0)-tf.expand_dims(X_col, 2)))
    outputs = - tf.reduce_sum(outputs, 1)

    def adder2d_grad(upstream):
        grad_W_col = tf.reduce_sum(
            (tf.expand_dims(X_col, 2)-tf.expand_dims(W_col, 0))
            * tf.expand_dims(upstream, 1),
            0)
        grad_W_col = grad_W_col / \
            tf.clip_by_value(
                tf.norm(grad_W_col, ord=2), 1e-12, tf.float32.max) * \
            tf.sqrt(1.0*W_col.shape[1]*W_col.shape[0]) / 5.0

        grad_X_col = tf.reduce_sum(
            -tf.clip_by_value((tf.expand_dims(X_col, 2) -
                               tf.expand_dims(W_col, 0)),
                              -1, 1)
            * tf.expand_dims(upstream, 1),
            2)

        return grad_X_col, grad_W_col

    return outputs, adder2d_grad


class Adder2D(Layer):
    def __init__(self,
                input_channel,
                output_channel,
                kernel_size,
                stride=1,
                padding=0,
                use_bias=False,
                trainable=True,
                kernel_initializer='glorot_uniform',  
                bias_initializer='zeros',
                automatic_differentiation=False,  
                **kwargs):
        super(Adder2D, self).__init__()

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.trainable = trainable

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        
        # Automatic differentiation
        self.automatic_differentiation = automatic_differentiation


    def build(self, input_shape):
        
        # kernel_shape: (h_filter, w_filter, in_channels, n_filters/out_channels)
        kernel_shape = [self.kernel_size, self.kernel_size, self.input_channel, self.output_channel]

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            trainable=self.trainable,
            dtype=tf.float32)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.output_channel),
                initializer=self.bias_initializer,
                trainable=self.trainable,
                dtype=tf.float32)
        else:
            self.bias = None

    # def set_kernel(self, kernel):
    #     self.kernel = tf.Variable(
    #             initial_value=kernel,
    #             trainable=False,
    #             dtype=tf.float32)


    def call(self, input_array):

        inputs, defined_kernel = input_array[0], input_array[1]
        self.kernel = defined_kernel

        # (h_filter, w_filter, in_channels, n_filters/out_channels)
        h_filter, w_filter, d_filter, n_filters = self.kernel.shape
        n_x, h_x, w_x, d_x = inputs.shape
        n_x = tf.shape(inputs)[0]  # actual shape

        # extract_patches takes a 4-D Tensor with shape [batch, in_rows, in_cols, depth] as input
        patches = tf.image.extract_patches(
            # .upper()
            inputs, sizes=[1, h_filter, w_filter, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME'
        )

        n_out, h_out, w_out, d_out = tf.shape(patches)[0], tf.shape(patches)[1], tf.shape(patches)[2], tf.shape(patches)[3]

        # reshape X_col and W_col for conv
        X_col = tf.reshape(patches, [-1, patches.shape[-1]])
        W_col = tf.reshape(self.kernel, [-1, n_filters])  # n_filters last

        # adder conv
        if self.automatic_differentiation:
            # Option 1, automatic differentiation
            outputs = tf.abs((tf.expand_dims(W_col, 0)-tf.expand_dims(X_col, 2)))
            outputs = - tf.reduce_sum(outputs, 1)
        else:
            # Option 2, custom gradient
            # print("X_col: ", X_col.shape)
            # print("W_col: ", W_col.shape)
            # print("input: ", inputs.shape)
            outputs = adder2d_im2col(X_col, W_col)

        # reshape outputs back
        # n_filters index last
        outputs = tf.reshape(outputs, [n_x, h_out, w_out, n_filters])


        if self.use_bias:
            outputs += self.bias[tf.newaxis, tf.newaxis, tf.newaxis, :]

        return outputs



def adder_func(inputs, kernel):


    # (h_filter, w_filter, in_channels, n_filters/out_channels)
    h_filter, w_filter, d_filter, n_filters = kernel.shape
    n_x, h_x, w_x, d_x = inputs.shape
    n_x = tf.shape(inputs)[0]  # actual shape

    # extract_patches takes a 4-D Tensor with shape [batch, in_rows, in_cols, depth] as input
    patches = tf.image.extract_patches(
        inputs, sizes=[1, h_filter, w_filter, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME'
    )


    n_out, h_out, w_out, d_out = tf.shape(patches)[0], tf.shape(patches)[1], tf.shape(patches)[2], tf.shape(patches)[3]
    # tf.print("patches.shape: ", tf.shape(patches))
    # reshape X_col and W_col for conv
    X_col = tf.reshape(patches, [-1, patches.shape[-1]])
    W_col = tf.reshape(kernel, [-1, n_filters])  # n_filters last

    # adder conv
    # if automatic_differentiation:
    #     # Option 1, automatic differentiation
    #     outputs = tf.abs((tf.expand_dims(W_col, 0)-tf.expand_dims(X_col, 2)))
    #     outputs = - tf.reduce_sum(outputs, 1)
    # else:
        # Option 2, custom gradient

    outputs = adder2d_im2col(X_col, W_col)

    # reshape outputs back
    # n_filters index last
    # print(tf.reshape(outputs, [-1, 1080, 1920, n_filters]).shape)
    outputs = tf.reshape(outputs, [n_out, h_out, w_out, n_filters])

    return outputs


