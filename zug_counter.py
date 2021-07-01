# USAGE
# python detect_circles.py --image game.JPG

# import the necessary packages
import numpy as np
import cv2 as cv
import functions as f
#import argpase

# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------

IMG_NAME = "game.JPG"
BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
BLACK = 4

# -----------------------------------------------------------
# Initialisation
# -----------------------------------------------------------

print("- Start...")
img_original = f.loadImage(IMG_NAME)
# cv.namedWindow("Result", cv.WINDOW_NORMAL) 

# -----------------------------------------------------------
# Process
# -----------------------------------------------------------

img_display, mask = f.extractColour(img_original, YELLOW)

# -----------------------------------------------------------
# Result
# -----------------------------------------------------------

result = np.concatenate((img_original, img_display), axis=1)

# show the output image
cv.imshow("Result", result)


print("- Finished !")

cv.waitKey(0)
cv.destroyAllWindows()
