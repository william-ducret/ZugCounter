# testing area

import numpy as np
import cv2 as cv
import functions as f
import random as rng
import functions as f

rng.seed(1)

#-------------------------------

img_original = f.loadImage("images/game_ch.JPG")

result = f.findBoard(img_original)

cv.imshow("result", result)

print("- Finished !")

cv.waitKey(0)
cv.destroyAllWindows()

