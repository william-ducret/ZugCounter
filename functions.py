
import numpy as np
import cv2 as cv

# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------

DISPLAY_WIDTH = 600
BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
BLACK = 4

# Low (min) and high (max) limit for each colour.
THRESHOLD_COLOUR = [
    ([140, 80, 0], [250, 200, 20]),
    ([0, 0, 0], [0, 0, 0]),
    ([0, 0, 130], [85, 85, 255]),
    ([0, 170, 200], [150, 255, 255]),
    ([0, 0, 0], [0, 0, 0])
]

# -----------------------------------------------------------
# Functions
# -----------------------------------------------------------

# load the image and resize it
def loadImage(img_name):
    
    img_original = cv.imread(img_name)
    h, w, c = img_original.shape
    img_ratio = h/w
    DISPLAY_HEIGHT = int(img_ratio * DISPLAY_WIDTH)

    # resizing
    img_original = cv.resize(img_original, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    return img_original


# extract a specific colour patern from the image (to find the train), return a mask
def shapeEnhancement(mask):

    # Morphologicial operations
    kernel_erode = np.ones((2,2),np.uint8)
    kernel_dilate = np.ones((2,2),np.uint8)
    mask = cv.erode(mask,kernel_erode,iterations = 1)
    mask = cv.dilate(mask, kernel_dilate, iterations= 1)
    mask = cv.erode(mask,kernel_erode,iterations = 1)
    mask = cv.dilate(mask, kernel_dilate, iterations= 1)
    mask = cv.erode(mask,kernel_erode,iterations = 1)
    mask = cv.dilate(mask, kernel_dilate, iterations= 1)

    return mask

# detect a colour according a range a BGR values, similar to extract_colour but more flexible and comfortable
def extractColour(img, colour):
    # loop over the boundaries
    # create NumPy arrays from the boundaries
    lower = np.array(THRESHOLD_COLOUR[colour][0], dtype = "uint8")
    upper = np.array(THRESHOLD_COLOUR[colour][1], dtype = "uint8")
        
    # find the colors within the specified boundaries to create a mask
    mask_raw = cv.inRange(img, lower, upper)
    # enhancement function
    mask = mask_raw #shapeEnhancement(mask_raw)
    # create the resulting image
    output = cv.bitwise_and(img, img, mask = mask)
    
#     result = np.concatenate((mask_raw, mask), axis=1)
#     cv.imshow("Mask", result)
    
    return output, mask

def reframe(img):
    
#     img = cv2.imread('C:/Users/Administrator/Desktop/testphoto/computer.jpg')
#     #Get source image width and height
#     w = img.shape[0]
#     h = img.shape[1]
#     #The quadrilateral coordinate points in the source image (The method of obtaining coordinate points can refer to my previous blog post)
#     point1 = np.array([[320,132],[1240,111],[414,800],[1351,738]],dtype = "float32")
#     #Get the coordinates of the rectangle after conversion
#     point2 = np.array([[0,0],[320,0],[0,180],[320,180]],dtype = "float32")
#     # point2 = np.array([[0,180],[320,180],[0,0],[320,0]],dtype = "float32")
#     M = cv2.getPerspectiveTransform(point1,point2)
#     out_img = cv2.warpPerspective(img,M,(w,h))
#     cv2.imshow("img",out_img)
#     cv2.waitKey(0)
    
    return img

def findBoard(img):
    

    
    return board
