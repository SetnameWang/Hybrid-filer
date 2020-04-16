import cv2
import numpy

def cross_correlation_2d(img, kernel):
    convB = _cross_correlation_2d(img[:, :, 0], kernel)
    convG = _cross_correlation_2d(img[:, :, 1], kernel)
    convR = _cross_correlation_2d(img[:, :, 2], kernel)
    return numpy.dstack([convB,convG,convR])
    
def _cross_correlation_2d(img, kernel):
    (height, width) = img.shape
    (kerHeight, kerWidth) = kernel.shape
    
    conv = numpy.zeros((height, width))
    
    # fill image edge with 0
    new1 = numpy.zeros((height, int(kerWidth / 2)), numpy.int)
    new2 = numpy.zeros((int(kerHeight / 2), width + new1.shape[1] * 2), numpy.int)
    
    temp = numpy.hstack([new1, numpy.hstack([img, new1])])
    temp = numpy.vstack([new2, numpy.vstack([temp, new2])])
    
    for i in range(height):
        for j in range(width):
            # calculate the value, value should between 0 and 255
            conv[i][j] = min(max(0, (temp[i:i + kerHeight, j:j + kerWidth] * kernel).sum()), 255)
    return conv
    
    
def convolve_2d(img, kernel):
    kernel = numpy.rot90(numpy.fliplr(kernel), 2)
    return cross_correlation_2d(img, kernel)
    
# create gaussian kernel
def gaussian_blur_kernel_2d(height, width, sigma):
    kernel = numpy.zeros((height, width))

    row = height / 2
    column = width / 2

    s = 2 * (sigma ** 2)

    for i in range(height):
        for j in range(width):
            x = i - row
            y = j - column
            kernel[i][j] = (1.0 / (numpy.pi * s)) * numpy.exp(-float(x ** 2 + y ** 2) / s)
    return kernel

def low_pass(img, height, width, sigma):
    kernel = gaussian_blur_kernel_2d(height, width, sigma)
    return convolve_2d(img,kernel)

def high_pass(img, height, width, sigma):
    return (img - low_pass(img, height, width, sigma))


img1 = cv2.imread("data/littledog.png")
img2 = cv2.imread("data/cat2.png")

ratio = 0.8
img1_res = low_pass(img1,13,13,4)
img2_res = high_pass(img2,15,15,7)
cv2.imwrite('left.jpg', img1_res)
cv2.imwrite('right.jpg', img2_res)

img_res = cv2.addWeighted(img1_res, ratio, img2_res, ratio, 0)
cv2.imwrite('hybrid.jpg', img_res)
