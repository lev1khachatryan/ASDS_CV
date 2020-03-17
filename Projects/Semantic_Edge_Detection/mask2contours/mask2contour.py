import numpy as np
import sys
import os
import cv2

def main():
    contour_path = r"C:\_Files\MyProjects\SemanticEdgeDetection\dataset\train\contours"
    mask_path = r"C:\_Files\MyProjects\SemanticEdgeDetection\dataset\train\masks"

    # iterate through the names of contents of the folder
    for image_path in os.listdir(mask_path):
        # Load the image and convert it to grayscale:
        image = cv2.imread(os.path.join(mask_path, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply cv2.threshold() to get a binary image
        ret, thresh = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

        # Find contours:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Get binar contour
        edge = np.zeros_like(image)
        cv2.drawContours(image=edge, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=1)
        edge = np.stack((edge,)*3, axis=-1)
        
        # Sav Edges
        cv2.imwrite(os.path.join(contour_path, image_path), edge)

if __name__ == '__main__':
    main()