'''
Defines the ResNet50, ResNet101 or ResNet152 model.
'''

import tensorflow as tf

l2 = tf.keras.regularizers.l2

class ResBlock(tf.keras.Model):
	# Defines the bottleneck resnet block that is used in ResNet 50 models and above.
	# I have also accounted for when the block needs to decrease the spatial dimensions of the feature maps through the
	# use of a projection shortcut.
	def __init__(self, num_filters, num_strides, proj_shortcut, weight_decay=1e-3, momentum=0.9, epsilon=0.0001):

		super(ResBlock, self).__init__()

		if proj_shortcut:
			self.conv_shortcut = tf.keras.layers.Conv2D(filters=num_filters[2], use_bias=False, kernel_size=1, activation=None, kernel_initializer="he_normal",
											kernel_regularizer=l2(weight_decay), strides=num_strides[0], padding="valid") # padding is "valid" because we need to decrease the spatial dimensions.

		# Set the strides of this conv layer to 2 if you want to downsample the feature maps 
		self.conv1 = tf.keras.layers.Conv2D(filters=num_filters[0], use_bias=False, kernel_size=1, activation=None, kernel_initializer="he_normal",
											kernel_regularizer=l2(weight_decay), strides=num_strides[0], padding="valid")

		self.bn1 = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)

		self.conv2 = tf.keras.layers.Conv2D(filters=num_filters[1], use_bias=False, kernel_size=3, activation=None, kernel_initializer="he_normal",
											kernel_regularizer=l2(weight_decay), strides=num_strides[1], padding="same")

		self.bn2 = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)

		self.conv3 = tf.keras.layers.Conv2D(filters=num_filters[2], use_bias=False, kernel_size=1, activation=None, kernel_initializer="he_normal",
											kernel_regularizer=l2(weight_decay), strides=num_strides[2], padding="valid") # can set padding to 'valid' since its a 1x1 conv.

		self.bn3 = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)


	def call(self, inputs, training): # Defines the forward propagation.
		net = self.conv1(inputs)
		net = self.bn1(net, training=training)
		net = tf.nn.relu(net)
		net = self.conv2(net)
		net = self.bn2(net, training=training)
		net = tf.nn.relu(net)
		net = self.conv3(net)
		net = self.bn3(net, training=training)

		# Add the inputs to the res block to the residual.
		if hasattr(self, "conv_shortcut"): 
			inputs = self.conv_shortcut(inputs)
		net = tf.add(inputs, net)

		net = tf.nn.relu(net)
		return net


class ResNet50_101_152(tf.keras.Model):
	# Defines the ResNet model.
	def __init__(self, num_outputs, resnet_model, weight_decay=1e-3):
		# "num_outputs": The number of output features of the ResNet model.
		# before doubling the number of filters and halving the spatial dimensions of the feature maps.
		# "resnet_model": Accepts either the integers 50, 101 or 152, denoting the ResNet model to construct.
		super(ResNet50_101_152, self).__init__()

		self.initial_conv = tf.keras.layers.Conv2D(filters=64, use_bias=False, kernel_size=7, activation=None, kernel_initializer="he_normal",
											kernel_regularizer=l2(weight_decay), strides=2, padding="valid")

		self.initial_pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")

		if resnet_model not in [50, 101, 152]:
			raise Exception("Please enter a valid ResNet model, either 50, 101 or 152.")

		# Defines the number of each type of ResBlock in each ResNet model.
		if resnet_model == 50:
			self.num_blocks = [3, 4, 6, 3]
		elif resnet_model == 101:
			self.num_blocks = [3, 4, 23, 3]
		elif resnet_model == 152:
			self.num_blocks = [3, 8, 36, 3]

		# resblocks with 64 filters.
		self.resblocks_64 = self._add_resblocks("res_64", [ResBlock([64, 64, 256], [1, 1, 1], True)] + \
														  [ResBlock([64, 64, 256], [1, 1, 1], False) for _ in range(self.num_blocks[0]-1)])

		# The first resblock after the number of filters doubles should have a projection shortcut, to ensure the depth of the input matches up.
		# This applies to resblocks 128 and onwards.
		self.resblocks_128 = self._add_resblocks("res_128", [ResBlock([128, 128, 512], [2, 1, 1], True)] + \
															[ResBlock([128, 128, 512], [1, 1, 1], False) for _ in range(self.num_blocks[1]-1)] 
															# TODO: Are the num of strides for the above 3 blocks correct??
												)

		self.resblocks_256 = self._add_resblocks("res_256", [ResBlock([256, 256, 1024], [2, 1, 1], True)] + \
															[ResBlock([256, 256, 1024], [1, 1, 1], False) for _ in range(self.num_blocks[2]-1)]
												)

		self.resblocks_512 = self._add_resblocks("res_512", [ResBlock([512, 512, 2048], [2, 1, 1], True)] + \
															[ResBlock([512, 512, 2048], [1, 1, 1], False) for _ in range(self.num_blocks[3]-1)]	
												)
		
		self.glo_avg_pool = tf.keras.layers.GlobalAveragePooling2D()

		self.outputs = tf.keras.layers.Dense(units=num_outputs)

	def _add_resblocks(self, name, blocks): 
		# Sets the REFERENCE of each ResBlock object as an attribute of the ResNet object.
		for i, block in enumerate(blocks):
			setattr(self, "{}_{}".format(name, i), block)

		return blocks

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

		return net