import tensorflow as tf

class ConvLayer(object):
  def __init__(self, filter_h, filter_w, pool_size, in_channels, out_channels,
               conv_activation=tf.nn.relu, stride=1
               ,scope="conv_layer"):
    self.filter_height = filter_h
    self.filter_width = filter_w
    self.pool_size = pool_size
    self.stride = stride
    self.in_channels = in_channels
    self.out_channels = out_channels

    self.scope = scope
    self.conv_activation = conv_activation

  def create_vars(self, reuse=False):
    with tf.compat.v1.variable_scope(self.scope, reuse=reuse):
      self.conv = tf.keras.layers.Conv2D(filters=self.out_channels,
                                          kernel_size=(self.filter_height, self.filter_width),
                                          strides=(self.stride, self.stride),
                                          padding='valid',
                                          data_format="channels_last",
                                          dilation_rate=(1, 1),
                                          activation=self.conv_activation,
                                          use_bias=True,
                                          kernel_initializer='glorot_uniform',
                                          bias_initializer='zeros',
                                          )
      self.pool = tf.keras.layers.MaxPool2D(pool_size=(self.pool_size, self.pool_size),
                                            strides=None,
                                            padding='valid',
                                            data_format=None)


  def apply(self, padded_input, reuse=tf.compat.v1.AUTO_REUSE, **kwargs):
    with tf.compat.v1.variable_scope(self.scope, reuse=reuse):
      tf.logging.info(padded_input)
      conv_out = self.conv(padded_input, **kwargs)

      outputs = self.pool(conv_out)

    return outputs



if __name__ == '__main__':
    image_h, image_w, image_ch = 12, 12, 1
    batch_size = 1
    inputs = tf.random.uniform((batch_size, image_h, image_w, image_ch), dtype=tf.float32)

    cnn = ConvLayer(filter_h=2, filter_w=2, pool_size=2, in_channels=1, out_channels=2)
    cnn.create_vars()
    outputs = cnn.apply(inputs)

    with tf.compat.v1.Session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      print(sess.run(outputs).shape)