import cv2
import numpy
import math
from convolution import convolution
from convolution import compare

def myEdgeFilter(img0, sigma):
    
    # smooth out the image
    hsize = 2 * math.ceil(3 * sigma) + 1
    kernal = cv2.getGaussianKernel(hsize, sigma)
    smoothImg = convolution(img0, kernal)
    
    # calculate x direction sobel
    kernal = numpy.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]))
    xSobel = convolution(smoothImg, kernal)
    cv2.imwrite("xSobel.png", xSobel)
    
    # calculate y direction sobel
    kernal = numpy.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]))
    ySobel = convolution(smoothImg, kernal)
    cv2.imwrite("ySobel.png", ySobel)
    
    
    # add x and y sobel
    # G(i,j) = sqrt(I^2(i,j) + I^2(i,j))
    sobel = numpy.sqrt(numpy.square(xSobel) + numpy.square(ySobel))
    cv2.imwrite("gradient magnitude.png", sobel)
    
    suppression = compare(xSobel, ySobel, sobel)
    cv2.imwrite("Non-Maximum Suppression.png", suppression)
