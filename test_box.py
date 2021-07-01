
import numpy as np
import cv2 as cv
import functions as f
import random as rng

rng.seed(1)

#-------------------------------

img_original = f.loadImage("game.JPG")

def findBoard(img):
    
    # conversion to gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # blurying, 3 methods
    gray_blurry = cv.blur(gray,(5,5))
    gray_blurry = cv.bilateralFilter(gray, 11, 17, 17)
    kernel_size = 9
    gray_blurry = cv.GaussianBlur(gray,(kernel_size, kernel_size),0)
    
    cv.imshow("gray_blurry", gray_blurry)
    
    flag, thresh = cv.threshold(gray_blurry, 50, 220, cv.THRESH_BINARY)
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


findBoard(img_original)

print("- Finished !")

cv.waitKey(0)
cv.destroyAllWindows()

