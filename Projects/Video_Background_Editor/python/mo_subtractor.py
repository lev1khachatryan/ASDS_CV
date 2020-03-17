from __future__ import print_function
import numpy as np
import cv2
import configparser

def main():
	# Find OpenCV version
	(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

	# Read config file
	config = configparser.ConfigParser()
	config.read('config.INI')
	# Get necessary information from config file
	INPUT_PATH = config['paths']['INPUT_PATH']
	OUTPUT_PATH = config['paths']['OUTPUT_PATH']
	BG_VIDEO_NAME = config['addition']['BG_VIDEO_NAME']

	# Open video and check if it successfully opened
	cap = cv2.VideoCapture(cv2.samples.findFileOrKeep(INPUT_PATH + BG_VIDEO_NAME))
	if not cap.isOpened:
	    print('Unable to open: ' + INPUT_PATH + BG_VIDEO_NAME)
	    exit(0)

	# Get fps and shape of the video
	if int(major_ver)  < 3:
		fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
	else:
		fps = cap.get(cv2.CAP_PROP_FPS)
	size = (int(cap.get(3)),int(cap.get(4))) 

	# Create video writer
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(OUTPUT_PATH + 'output.avi', fourcc, fps, size)

	# Create background subtractor
	backSub = cv2.createBackgroundSubtractorMOG2()
	# backSub = cv2.createBackgroundSubtractorKNN()

	while True:
	    _, frame = cap.read()
	    if frame is None:
	    	break

	    fgMask = backSub.apply(frame)

	    fgMask = cv2.cvtColor(fgMask,cv2.COLOR_GRAY2RGB)
	    # fgMask = np.stack([fgMask]*3, axis = -1)
	    # out.write(fgMask * frame)
	    out.write(fgMask)

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
    main()