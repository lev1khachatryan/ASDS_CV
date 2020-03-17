# USAGE
# python bg_fg_combiner.py

from __future__ import print_function
import numpy as np
import cv2
import configparser
import os

# Find OpenCV version
(MAJOR_VER, MINOR_VER, SUBMINOR_VER) = (cv2.__version__).split('.')

def bgSubtractor(path):
	'''
	Estimate and extract the background of a scene when the camera is static 
	and there are some moving objects in the scene.
	------------------------
	Parameters
		path (string): path of the video from which we want to extract the background
	Returns
		nd.array: background of the video
	------------------------
	'''

	# Open the video and check if it successfully done
	cap = cv2.VideoCapture(cv2.samples.findFileOrKeep(path))
	if not cap.isOpened:
	    print('Unable to open: ' + path)
	    exit(0)

	# Randomly select 25 frames
	frame_ids = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
	
	# Store selected frames in an array
	frames = []
	for fid in frame_ids:
	    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
	    ret, frame = cap.read()
	    frames.append(frame)
	
	# Calculate the median along the time axis
	median_frame = np.median(frames, axis=0)
	# cv2.imwrite('background.jpg', median_frame)
	
	# When everything done, cleanup the camera and close any open windows
	cap.release()
	cv2.destroyAllWindows()

	return median_frame


def fgBgCombiner(path, bg_frame, save_dir):
	'''
	Change background in videos using GrabCut algorithm
	------------------------
	Parameters
		path (string): path of the video to change the background
		bg_frame (array like): background image (or nd.array)
		save_dir (string): path where to save obtained video
	Returns
		None
	------------------------
	'''

	# Get video name from full path
	file_name = path.split('/')[-1].split('.')[0]

	# Open the video and check if it successfully done
	cap = cv2.VideoCapture(cv2.samples.findFileOrKeep(path))
	if not cap.isOpened:
	    print('Unable to open: ' + path)
	    exit(0)

	# Get fps and shape of the video
	if int(MAJOR_VER) < 3:
		fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
	else:
		fps = cap.get(cv2.CAP_PROP_FPS)
	size = (int(cap.get(3)),int(cap.get(4)))

	# Create video writer
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(save_dir + file_name + '.avi', fourcc, fps, size)

	# Resize background frame (image) to video shape
	background = cv2.resize(bg_frame, size, interpolation=cv2.INTER_AREA)
	
	# Loop over the frames of the video
	while True:
		# Grab the current frame
	    _, frame = cap.read()
	    # If the frame could not be grabbed, then we have reached the end of the video
	    if frame is None:
	    	break

	    # Convert it to grayscale
	    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	    # Initial mask
	    mask = np.zeros(frame.shape[:2], np.uint8)
	    # These are arrays used by the algorithm internally.
	    bgdModel = np.zeros((1, 65), np.float64)
	    fgdModel = np.zeros((1, 65), np.float64)
	    # Specify a region of interest (RoI) and apply grabCut algorithm
	    rect = (200, 50, 300, 400)
	    # Number of iterations the algorithm should run is 1
	    # which is fast but not good for correct segmentation
	    cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
	    # New mask for moving object
	    mask2 = np.where((mask == 2) | (mask == 0), (0,), (1,)).astype('uint8')
	    frame = frame * mask2[:, :, np.newaxis]
	    mask_1 = frame > 0
	    mask_2 = frame <= 0
	    # Linear combination of bgd and fgd frames with mask_1 and mask_2 "scaliars"
	    combination = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) * mask_1 + background * mask_2
	    combination = combination.astype(dtype=np.uint8)
	    # Write combined frame
	    out.write(combination)

	# When everything done, cleanup the camera and close any open windows
	print('Background change is finished')
	cap.release()
	cv2.destroyAllWindows()


def main():

	# Read config file
	config = configparser.ConfigParser()
	config.read('config.INI')
	# Get necessary information from config file
	INPUT_PATH = config['paths']['INPUT_PATH']
	OUTPUT_PATH = config['paths']['OUTPUT_PATH']
	BG_VIDEO_NAME = config['addition']['BG_VIDEO_NAME']

	background = bgSubtractor(INPUT_PATH + BG_VIDEO_NAME)

	for file in os.listdir(INPUT_PATH):
	    if file.endswith('.mp4'):
	        path=os.path.join(INPUT_PATH, file)
	        fgBgCombiner(path, background, OUTPUT_PATH)

if __name__ == '__main__':
    main()