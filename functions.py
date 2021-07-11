
import numpy as np
import cv2 as cv
import random as rng

rng.seed(1)

# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------

DISPLAY_WIDTH = 600
BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
BLACK = 4
FRAME = 5

# Low (min) and high (max) limit for each colour, BGR.
THRESHOLD_COLOUR = [
    ([140, 80, 0], [250, 200, 20]),
    ([0, 0, 0], [0, 0, 0]),
    ([0, 0, 130], [85, 85, 255]),
    ([0, 170, 200], [150, 255, 255]),
    ([0, 0, 0], [0, 0, 0]),
     ([0, 0, 3], [25, 25, 65])
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
    # create NumPy arrays from the boundaries
    lower = np.array(THRESHOLD_COLOUR[colour][0], dtype = "uint8")
    upper = np.array(THRESHOLD_COLOUR[colour][1], dtype = "uint8")
        
    # find the colors within the specified boundaries to create a mask
    mask_raw = cv.inRange(img, lower, upper)
    # enhancement function
    mask = mask_raw #shapeEnhancement(mask_raw)
    # create the resulting image
    output = cv.bitwise_and(img, img, mask = mask)
    
    return output, mask

def extractBoard(img):
    
    # blurying
    blurry = cv.GaussianBlur(img,(5, 5),5)
    
    # create NumPy arrays from the boundaries
    lower = np.array(THRESHOLD_COLOUR[FRAME][0], dtype = "uint8")
    upper = np.array(THRESHOLD_COLOUR[FRAME][1], dtype = "uint8")
        
    # find the colors within the specified boundaries to create a mask
    mask_raw = cv.inRange(img, lower, upper)
    # enhancement function
    mask = mask_raw #shapeEnhancement(mask_raw)
    # create the resulting image
    output = cv.bitwise_and(img, img, mask = mask)

    # Enhancement with morphologicial operations
    kernel_erode = np.ones((2,2),np.uint8)
    kernel_dilate = np.ones((4,4),np.uint8)
    mask = cv.erode(mask,kernel_erode,iterations = 1)
    mask = cv.dilate(mask, kernel_dilate, iterations= 1)
    mask = cv.erode(mask,kernel_erode,iterations = 2)
    mask = cv.dilate(mask, kernel_dilate, iterations= 5)
    
    cv.imshow("mask", mask)
    
    # line detection
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 30  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 180  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv.HoughLinesP(mask, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    
    cv.imshow("line_image", line_image)
    
    # Draw the lines on the  image
    lines_edges = cv.addWeighted(img, 0.8, line_image, 1, 0)
    cv.imshow("lines_edges", lines_edges)
    
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
    
    # conversion to gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # blurying
    gray_blurry = cv.bilateralFilter(gray, 11, 21, 7)
    gray_blurry = cv.GaussianBlur(gray,(7, 7),13)
    
    cv.imshow("gray_blurry", gray_blurry)
    
    flag, thresh = cv.threshold(gray_blurry, 50, 255, cv.THRESH_BINARY)
    thresh = cv.adaptiveThreshold(gray_blurry, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 59, 20)
    
#     flag, thresh = cv.threshold(gray_blurry, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    
    cv.imshow("thresh", thresh)
    
    # Canny edge detector
    edges = cv.Canny(gray_blurry, 100, 200)
    
#     cv.imshow("Gray", gray)
#     cv.imshow("Blurry", gray_blurry)
    cv.imshow("Canny", edges)
    
    # line detection
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 30  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 130  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    
    # Draw the lines on the  image
    lines_edges = cv.addWeighted(img, 0.8, line_image, 1, 0)
    
    cv.imshow("line_image", line_image)
    cv.imshow("lines_edges", lines_edges)
    
    # find contours
    contours, hierarchy = cv.findContours(edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # all contours displaying
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(img, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
    
    cv.imshow("Contours", img)
    
    #     # select the 10 largest contours
#     contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]
#     
#     # displaying the contours
#     cv.drawContours(img, contours, -1, (0, 255, 0), 1)

    screenCnt = None
    
    board = img
    
    return board
