# This script helps to discard the "bad" images

from __future__ import print_function
import configparser
from utils import *

if __name__ == '__main__':
	config = configparser.ConfigParser()
	config.read('config.INI')

	DATA_DIR = config['paths']['DATA_DIR']
	TEST_DIR = config['paths']['TEST_DIR']

	t0 = datetime.now()

	discard_bad_images(TEST_DIR, DATA_DIR, list_test_images(TEST_DIR))

	print('Elapsed time: %s\n' % (datetime.now() - t0))

