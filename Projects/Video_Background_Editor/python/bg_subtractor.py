from __future__ import print_function
import numpy as np
import cv2
import configparser

def main():
	# Read config file
	config = configparser.ConfigParser()
	config.read('config.INI')
	# Get necessary information from config file
	INPUT_PATH = config['paths']['INPUT_PATH']
	OUTPUT_PATH = config['paths']['OUTPUT_PATH']
	BG_VIDEO_NAME = config['addition']['BG_VIDEO_NAME']

	# Open Video
	cap = cv2.VideoCapture(cv2.samples.findFileOrKeep(INPUT_PATH + BG_VIDEO_NAME))

	if not cap.isOpened:
	    print('Unable to open: ' + INPUT_PATH + BG_VIDEO_NAME)
	    exit(0)

	# Randomly select 25 frames
	frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
	 
	# Store selected frames in an array
	frames = []
	for fid in frameIds:
	    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
	    ret, frame = cap.read()
	    frames.append(frame)
	 
	# Calculate the median along the time axis
	medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

	# Display median frame
	# cv2.imshow('frame', medianFrame)
	# cv2.waitKey(0)

	# Save median frame to output folder
	cv2.imwrite(OUTPUT_PATH + 'background.jpg', medianFrame)
	
	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
    main()