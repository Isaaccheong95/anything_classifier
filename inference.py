import tensorflow as tf
import os
import cv2
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from resnet_18_34 import*
from resnet50_101_152 import *

'''
Labels:
Bird - 0
Cats - 1
Dogs - 2
Fish - 3
'''
label_dict = {0:"Bird", 1:"Cat", 2:"Dog", 3:"Fish"}

tf.enable_eager_execution()
print(tf.__version__)

INSIZE = 224

TEST_FLDR = "./test_infer/" # Folder containing the iamges you want to perform inference on.
LOAD_WEIGHTS_FROM = "lr_0.001epochs_50resnet_18"

tfe = tf.contrib.eager

checkpoint_dir = os.path.join("./tboard_logs", LOAD_WEIGHTS_FROM)

# Read in the saved model hyperparameters.
model_args = eval(open(os.path.join(checkpoint_dir, "model_hyperparams.txt"), "r").read())

if LOAD_WEIGHTS_FROM[-2:] in ["18", "34"]:
	model = ResNet18_34(**model_args)
else:
	model = ResNet50_101_152(**model_args)

root = tfe.Checkpoint(model=model,
					  optimizer_step=tf.train.get_or_create_global_step())

root.restore(tf.train.latest_checkpoint(checkpoint_dir))
print("Loaded in model weights from {}".format(checkpoint_dir))

for img in os.listdir(TEST_FLDR):
	test_img = cv2.resize(cv2.imread(os.path.join(TEST_FLDR, img), cv2.IMREAD_UNCHANGED), (INSIZE, INSIZE))
	test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB) / 255.

	test_input = tf.cast(np.reshape(test_img, (-1, INSIZE, INSIZE, 3)), tf.float32)

	pred = tf.nn.softmax(model(test_input, training=False))
	print("Image: {}".format(img))
	print("Model output: {}".format(pred.numpy()))
	print("Final prediction: {}".format(label_dict[np.argmax(pred.numpy(), axis=1)[0]]))
	print()
