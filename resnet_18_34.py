'''
Defines the ResNet 18 and ResNet 34 models.
'''

import tensorflow as tf

l2 = tf.keras.regularizers.l2

class ResBlock(tf.keras.Model):
	# Defines the full pre-activation version of a residual block.
	def __init__(self, num_filters, proj_shortcut, stride=1, weight_decay=1e-3, momentum=0.9, epsilon=0.0001):

		super(ResBlock, self).__init__()

		if stride > 1:
			padding = "valid"
		elif stride == 1:
			padding = "same"

		if proj_shortcut:
			# This conv layer will have a stride of 2 whenever we double the number of filters.
			self.conv_shortcut = tf.keras.layers.Conv2D(filters=num_filters[0], use_bias=False, kernel_size=1, activation=None, kernel_initializer="he_normal",
											kernel_regularizer=l2(weight_decay), strides=stride, padding=padding) # padding is "valid" because we need to decrease the spatial dimensions.

		self.bn1 = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)

		# This conv layer will have a stride of 2 whenever we double the number of filters.
		self.conv1 = tf.keras.layers.Conv2D(filters=num_filters[0], use_bias=False, kernel_size=3, activation=None, kernel_initializer="he_normal",
											kernel_regularizer=l2(weight_decay), strides=stride, padding=padding)

		self.bn2 = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)

		self.conv2 = tf.keras.layers.Conv2D(filters=num_filters[1], use_bias=False, kernel_size=3, activation=None, kernel_initializer="he_normal",
											kernel_regularizer=l2(weight_decay), strides=1, padding="same")

	def call(self, inputs, training):
		net = self.bn1(inputs)
		net = tf.nn.relu(net)

		if hasattr(self, "conv_shortcut"):
			inputs = self.conv_shortcut(inputs)
			# have to zero pad with a padding of 1 whenever we downsample the spatial dimensions.
			net = tf.image.resize_image_with_crop_or_pad(net, int(net.shape[2])+2, int(net.shape[1])+2)

		net = self.conv1(net)
		net = self.bn2(net)
		net = tf.nn.relu(net)
		net = self.conv2(net)


		net = tf.add(inputs, net)
		return net

class ResNet18_34(tf.keras.Model):
	# Defines the ResNet model.
	def __init__(self, num_outputs, resnet_model, weight_decay=1e-3):
		# "num_outputs": The number of output features of the ResNet model.
		# before doubling the number of filters and halving the spatial dimensions of the feature maps.
		# "resnet_model": Accepts either the integers 50, 101 or 152, denoting the ResNet model to construct.

		super(ResNet18_34, self).__init__()

		self.initial_conv = tf.keras.layers.Conv2D(filters=64, use_bias=False, kernel_size=7, activation=None, kernel_initializer="he_normal",
											kernel_regularizer=l2(weight_decay), strides=2, padding="valid")

		self.initial_pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")

		if resnet_model not in [18, 34]:
			raise Exception("Please enter a valid ResNet model, either 18 or 34.")

		# Defines the number of each type of ResBlock in each ResNet model.
		if resnet_model == 18:
			self.num_blocks = [2, 2, 2, 2]
		elif resnet_model == 34:
			self.num_blocks = [3, 4, 6, 3]

		self.resblocks_64 = [ResBlock([64, 64], False) for _ in range(self.num_blocks[0])]

		self.resblocks_128 = [ResBlock([128, 128], True, 2)] + [ResBlock([128, 128], False) for _ in range(self.num_blocks[1]-1)]

		self.resblocks_256 = [ResBlock([256, 256], True, 2)] + [ResBlock([256, 256], False) for _ in range(self.num_blocks[2]-1)]

		self.resblocks_512 = [ResBlock([512, 512], True, 2)] + [ResBlock([512, 512], False) for _ in range(self.num_blocks[3]-1)]

		self.glo_avg_pool = tf.keras.layers.GlobalAveragePooling2D()

		self.outputs = tf.keras.layers.Dense(units=num_outputs)

#	def _add_resblocks(self, name, blocks):
#		for i, block in enumerate(blocks):
#			setattr(self, "{}_{}".format(name, i), block)

#		return blocks

	@tf.contrib.eager.defun
	def call(self, inputs, training):
		# need to first zero pad the input with a padding of 3, s.t. the size becomes 230x230.
		inputs = tf.image.resize_image_with_crop_or_pad(inputs, 230, 230)

		net = self.initial_conv(inputs)
		# print("after initial conv:", net.shape)
		net = self.initial_pool(net)
		# print("after initial pool:", net.shape)

		for rb_64 in self.resblocks_64:
			net = rb_64(net, training=training)
		# print("after rb_64:", net.shape)

		for rb_128 in self.resblocks_128:
			net = rb_128(net, training=training)
		# print("after rb_128:", net.shape)

		for rb_256 in self.resblocks_256:
			net = rb_256(net, training=training)
		# print("after rb_256:", net.shape)

		for rb_512 in self.resblocks_512:
			net = rb_512(net, training=training)
		# print("after rb_512:", net.shape)

		net = self.glo_avg_pool(net)
		net = self.outputs(net)
		# outputs logit values

		return net
