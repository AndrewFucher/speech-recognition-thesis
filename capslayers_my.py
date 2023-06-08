import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

# Primary Capsule Layer
class PrimaryCap(layers.Layer):
    def __init__(self, dim_capsule, n_channels, kernel_size, strides, padding):
        super(PrimaryCap, self).__init__()
        self.dim_capsule = dim_capsule
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        self.conv2d = layers.Conv1D(filters=self.dim_capsule * self.n_channels,
                                    kernel_size=self.kernel_size,
                                    strides=self.strides,
                                    padding=self.padding)
        self.built = True

    def call(self, inputs):
        output = self.conv2d(inputs)
        output = layers.Reshape(target_shape=(-1, self.dim_capsule))(output)
        return layers.Lambda(squash)(output)

# Capsule Layer
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        _, self.input_num_capsule, self.input_dim_capsule = input_shape
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule, self.dim_capsule, self.input_dim_capsule],
                                 initializer='glorot_uniform',
                                 name='W')
        self.built = True

    def call(self, inputs):
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 1), -1)
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1, 1])
        inputs_hat = tf.squeeze(tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled))
        b = tf.zeros(shape=[tf.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            outputs = squash(tf.matmul(c, inputs_hat))
            if i < self.routings - 1:
                b += tf.matmul(outputs, inputs_hat, transpose_a=True)
        return outputs

# Squash Activation Function
def squash(vector):
    norm = tf.norm(vector, axis=-1, keepdims=True)
    norm_squared = norm * norm
    return (vector / norm) * (norm_squared / (1 + norm_squared))

# Define the Capsule Network model
def CapsNet2(input_shape, num_classes, num_capsule, batch_size=8, dim_capsule=8, routings=3):
    x = layers.Input(shape=input_shape)

    # Conv1D layer
    conv1 = layers.Conv1D(filters=256, kernel_size=9, strides=1, padding='same', activation='relu')(x)

    # PrimaryCapsule layer
    primarycaps = PrimaryCap(dim_capsule=dim_capsule, n_channels=8, kernel_size=9, strides=2, padding='same')(conv1)

    # DigitCapsule layer
    digitcaps = CapsuleLayer(num_capsule=num_capsule, dim_capsule=dim_capsule, routings=routings)(primarycaps)

    # Flatten the DigitCaps output for further processing
    digitcaps_flatten = layers.Flatten()(digitcaps)

    # Output layer for text classification
    output = layers.Dense(num_classes, activation='softmax')(digitcaps_flatten)

    # Create the model
    model = models.Model(inputs=x, outputs=output)

    return model