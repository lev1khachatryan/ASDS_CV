from __future__ import print_function
import numpy as np
import random
import configparser
from utils import *

if __name__ == '__main__':
	config = configparser.ConfigParser()
	config.read('config.INI')

	DATA_DIR = config['paths']['DATA_DIR']
	PERCENT_OF_VALSET = float(config['other']['PERCENT_OF_VALSET'])

	t0 = datetime.now()

	london_imgs = list_test_images(DATA_DIR + 'train/london/')
	yerevan_imgs = list_test_images(DATA_DIR + 'train/yerevan/')

	move_to_validation_london  = random.sample(london_imgs, int(np.ceil(len(london_imgs)*PERCENT_OF_VALSET)))
	move_to_validation_yerevan = random.sample(yerevan_imgs, int(np.ceil(len(yerevan_imgs)*PERCENT_OF_VALSET)))

	validation_path = create_dir('validation', DATA_DIR)
	yerevan_validation_path = create_dir('yerevan', validation_path)
	london_validation_path  = create_dir('london', validation_path)

	for path in move_to_validation_london:
		shutil.move(path, london_validation_path)

	for path in move_to_validation_yerevan:
		shutil.move(path, yerevan_validation_path)

	print('Elapsed time: %s\n' % (datetime.now() - t0))

