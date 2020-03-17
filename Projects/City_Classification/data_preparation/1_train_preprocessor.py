# This script helps to discard the "bad" images

from __future__ import print_function
import configparser
from utils import *

if __name__ == '__main__':
	config = configparser.ConfigParser()
	config.read('config.INI')

	DATA_DIR = config['paths']['DATA_DIR']

	t0 = datetime.now()

	discard_bad_images(DATA_DIR, DATA_DIR, list_train_images(DATA_DIR))

	# After deleting all corrupted images, let's move correct ones to appropriate folders.
	train_path   = create_dir('train', DATA_DIR)
	yerevan_path = create_dir('yerevan', train_path)
	london_path  = create_dir('london', train_path)

	paths = list_train_images(DATA_DIR)
	for path in paths:
		if 'london' in path:
			shutil.move(path, london_path)
		elif 'yerevan' in path:
			shutil.move(path, yerevan_path)

	print('Elapsed time: %s\n' % (datetime.now() - t0))

