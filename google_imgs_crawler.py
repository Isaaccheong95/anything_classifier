'''
Downloads images from Google Images.
'''

import csv
import os
from tqdm import tqdm
from google_images_download import google_images_download
from os.path import isdir, join
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--query", "-q", default=None)
parser.add_argument("--numberDownload", "-n", default=None, help="Number of images to download")

args = parser.parse_args()

response = google_images_download.googleimagesdownload()

output_dir = "./anythingClassifier/dataset/train/"

queries = args.query

# Can name the folders as the celebrity name instead by changing "filename" to queries.split(",")[0] and vice-versa
if not isdir(join(output_dir,queries)):
	os.mkdir(join(output_dir,queries))

# print(url_filename)
# The 'keywords' argument is a string of queries that are separated by commas.
arguments = {"keywords":queries, "limit":args.numberDownload, "output_directory": output_dir, "print_urls":True, \
"format":"jpg", "chromedriver":"C:\\Users\\isaac\\Desktop\\chromedriver_win32\\chromedriver.exe"}

response.download(arguments)

