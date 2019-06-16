'''
Preprocesses the images and stores them in a .npy file with their corresponding labels and file paths.
'''

from tqdm import tqdm
import os
import random
import numpy as np
from random import shuffle
import cv2

'''
Labels:
Bird - 0
Cats - 1
Dogs - 2
Fish - 3
'''

INSIZE = 224

TRAIN_IMGS_LOCS = ["./dataset/train/bird/", "./dataset/train/cats/", "./dataset/train/dogs/", "./dataset/train/fish/"]
VAL_IMGS_LOCS = ["./dataset/val/bird/", "./dataset/val/cats/", "./dataset/val/dogs/", "./dataset/val/fish/"]
TEST_IMGS_LOCS = ["./dataset/test/bird/", "./dataset/test/cats/", "./dataset/test/dogs/", "./dataset/test/fish/"]


def create_data(img_location, train_data=True):
	'''
	Arguments:
	img_location: Pass in a list with the paths to the folders of positive and negative images.

	Returns:
	3-element lists. Each 3-element list consists of a numpy array representing the image, a numeric label, and lastly
	the path of the image, in this order.
	'''
	data = []

	for label, img_loc in enumerate(img_location):

		img_list = os.listdir(img_loc)

		#shuffle(img_list)
		print("in {}...".format(img_loc))

		for img in tqdm(img_list):
			path = os.path.join(img_loc, img)

			colorImg = cv2.resize(cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), (INSIZE, INSIZE)) # Here, we resize the images to 224 x 224.
			data.append([colorImg, label, path])

	shuffle(data) # We shuffle up the images.

	if "train" in img_location[0]:
		np.save("./dataset/train_data-{}.npy".format(INSIZE), data) # I will save the results so that next time I won't have to rerun this function.
	elif "val" in img_location[0]:
		np.save("./dataset/val_data-{}.npy".format(INSIZE), data)
	else:
		np.save("./dataset/test_data-{}.npy".format(INSIZE), data)

	return data

if __name__ == "__main__":
	create_data(TRAIN_IMGS_LOCS)
	create_data(VAL_IMGS_LOCS)
	create_data(TEST_IMGS_LOCS)
