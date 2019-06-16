import tensorflow as tf
from os.path import join, isdir
import os
from lr_schedules import *
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import argparse
import cv2
from resnet_18_34 import*
from resnet50_101_152 import *

tf.enable_eager_execution()
print(tf.__version__)

INSIZE = 224 # Input image dimensions for ResNet models.
NUM_CHANNELS = 3 # MNIST is grayscale, change to 3 if your images are RGB.

tfe = tf.contrib.eager

parser = argparse.ArgumentParser()

parser.add_argument("--logDir", "-l", default="tboard_logs", help="Path of directory where you will save your tensorflow summaries.")
parser.add_argument("--lr", "-m", type=float, default=1e-3, help="Max LR in cyclical learning rate schedule.")
parser.add_argument("--batchSize", "-b", type=int, default=16)
parser.add_argument("--L2reg", "-r", type=float, default=5e-2, help="Lambda value for L2 regularization.")
parser.add_argument("--numOutputs", "-n", type=int, default=4, help="Number of classes in your dataset.")
parser.add_argument("--numEpochs", "-e", type=int, default=5)
parser.add_argument("--lr_schedule", "-s", default="exp_decay", help="Takes in the function names \
of the different learning rate schedules. Refer to the lr_schedules.py file for the function names.")

# Command line args for ResNet
parser.add_argument("--resModel", "-o", type=int, default=18, help="Integer value indicating the ResNet model you want to use, either 18, 34, 50, 101 or 152.")

args = parser.parse_args()

model_settings = {
					'num_outputs': args.numOutputs,
					'weight_decay': args.L2reg,
					'resnet_model': args.resModel
}

if args.resModel in [18, 34]:
	model = ResNet18_34(**model_settings)
elif args.resModel in [50, 101, 152]:
	model = ResNet50_101_152(**model_settings)
else:
	raise Exception("Please enter a valid ResNet model, either 18, 34, 50, 101 or 152.")

train_data = np.load("./dataset/train_data-224.npy")
val_data = np.load("./dataset/val_data-224.npy")
test_data = np.load("./dataset/test_data-224.npy")

print("Number training images:", len(train_data))
print("Number validation images:", len(val_data))
print("Number testing images:", len(test_data))

epoch_length = len(train_data) // args.batchSize # number of iterations needed per epoch
epochs = args.numEpochs

if args.lr_schedule == "cyclical_lr":
	stepsize = 2 * epoch_length # thus, since (2 * stepsize) is equal to 1 cycle in the cyclical learning rate schedule, 4 epochs would equal to 1 cycle.
	base_lr = args.lr / 3
	get_lr = cyclical_lr(base_lr=base_lr, max_lr=args.lr,
										stepsize=stepsize,
										decrease_lr_by=0.15)
else:
	base_lr = args.lr
learning_rate_tf = tfe.Variable(base_lr)

# Directory to save the tensorflow summaries for this particular training.
checkpoint_dir = join(args.logDir, "lr_"+str(args.lr)+"epochs_"+str(args.numEpochs)+"resnet_"+str(args.resModel))
if not isdir(checkpoint_dir):
	os.mkdir(checkpoint_dir)
# Save the model's hyperparameters so we can initialize the model in the same way.
with open(join(checkpoint_dir, "model_hyperparams.txt"), "w") as f:
	f.write(str(model_settings))

# Name for the saved weights files for the model.
checkpoint_prefix = join(checkpoint_dir, "ckpt")

# Initialize the summary writers
train_writer = tf.contrib.summary.create_file_writer(join(checkpoint_dir, "train"))
val_writer = tf.contrib.summary.create_file_writer(join(checkpoint_dir, "val"))

# IMPT: Comment out your chosen optimizer.
# Different optimization algorithms to try.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_tf, beta1=0.9, beta2=0.99)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_tf)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate_tf)
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_tf, momentum=0.9)

root = tfe.Checkpoint(optimizer=optimizer,
					  model=model,
					  optimizer_step=tf.train.get_or_create_global_step())

print("Using model ResNet-{}...".format(args.resModel))
# Training loop.
for epoch in range(epochs):
	print("Training epoch {} out of {}...".format(epoch+1, epochs))
	trng_loss = 0

	with train_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):

		for iteration in tqdm(range(epoch_length)):
			batch_x = []
			batch_y = []

			for i in range(args.batchSize):
				batch_x.append(train_data[iteration*args.batchSize+i][0] / 255.)
				batch_y.append(train_data[iteration*args.batchSize+i][1])

			batch_x = np.reshape(batch_x, (args.batchSize, INSIZE, INSIZE, NUM_CHANNELS))
			batch_x = tf.cast(batch_x, tf.float32) # Otherwise will cause: AttributeError: 'tuple' object has no attribute 'ndims'

			batch_y = tf.one_hot(batch_y, depth=args.numOutputs)

			with tf.GradientTape() as tape:
				preds = model(batch_x, training=True)
				# print("preds.shape", preds.shape)
				loss = tf.losses.softmax_cross_entropy(batch_y, preds)
				trng_loss += loss

				tf.contrib.summary.scalar('loss', loss)
				tf.contrib.summary.scalar('learning_rate', learning_rate_tf)

			# compute grads w.r.t model parameters and update weights
			grads = tape.gradient(loss, model.trainable_variables)
			optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=tf.train.get_or_create_global_step())

			# Train using the chosen lr schedule.
			if args.lr_schedule == "cyclical_lr":
				new_lr = next(get_lr)
				# print("cyclical_lr")
			elif args.lr_schedule == "exp_decay":
				new_lr = exp_decay(base_lr, epoch)
				# print("exp_decay")
			elif args.lr_schedule == "step_decay":
				new_lr = step_decay(base_lr, epoch)
				# print("step_decay")
			learning_rate_tf.assign(new_lr)

		print("Training loss:", (trng_loss/len(train_data)).numpy())

		# Model evaluation.
		val_preds = []
		softmax_pred = []
		val_labels = []
		print("Evaluating model...")

		for i, val_eg in enumerate(val_data):
			val_input = tf.cast(np.expand_dims(val_eg[0] / 255., 0), tf.float32)
			pred = model(val_input, training=False)[0].numpy()
			if i < 10:
				print(tf.nn.softmax(pred), val_eg[1])
			val_preds.append(pred)
			softmax_pred.append(tf.nn.softmax(pred).numpy())
			val_labels.append(val_eg[1])

		val_preds = np.array(val_preds)
		softmax_pred = np.array(softmax_pred)
		oh_val_labels = tf.one_hot(val_labels, depth=args.numOutputs)

		val_loss = tf.losses.softmax_cross_entropy(oh_val_labels, val_preds)
		val_loss /= len(val_data) # find the avg validation loss.

		val_acc = np.mean(np.argmax(softmax_pred, axis=1) == val_labels)
		print("Validation accuracy:", val_acc)
		print("Validation loss:", val_loss.numpy())

		# find the AUC score.
		# fpr, tpr, thresholds = metrics.roc_curve(val_labels, val_preds, pos_label=1)
		# print("AUC score: {}".format(metrics.auc(fpr, tpr)))
		if epoch % 5 == 0:
			root.save(file_prefix=checkpoint_prefix)
			print("Model saved.")

		with val_writer.as_default(), tf.contrib.summary.always_record_summaries():
			tf.contrib.summary.scalar('val loss', val_loss, step=epoch+1)
			# tf.contrib.summary.scalar('val AUC', metrics.auc(fpr, tpr), step=epoch+1)
			tf.contrib.summary.scalar('val accuracy', val_acc, step=epoch+1)

# Model Testing
test_preds = []
test_labels = []
softmax_pred_test = []
print("Testing model...")

for test_eg in tqdm(test_data):
	test_input = tf.cast(np.expand_dims(test_eg[0] / 255., 0), tf.float32)
	t_pred = model(test_input, training=False)[0].numpy()
	test_preds.append(t_pred)
	softmax_pred_test.append(tf.nn.softmax(t_pred).numpy())
	test_labels.append(test_eg[1])

test_preds = np.array(test_preds)
softmax_pred_test = np.array(softmax_pred_test)
oh_test_labels = tf.one_hot(test_labels, depth=args.numOutputs)

test_loss = tf.losses.softmax_cross_entropy(oh_test_labels, test_preds)
test_loss /= len(test_data) # find the avg test loss.

test_acc = np.mean(np.argmax(softmax_pred_test, axis=1) == test_labels)
print("Test accuracy:", test_acc)
print("Test loss:", test_loss.numpy())
