import cv2
import numpy

def convolution(img, kernel):
    '''
    # eivided into three channels
    convB = _convolution(img[:,:,0],kernel)
    convG = _convolution(img[:,:,1],kernel)
    convR = _convolution(img[:,:,2],kernel)
    
    return numpy.dstack([convB,convG,convR])
    '''
    return _convolution(img, kernel)

def compare(imgX, imgY, img):
    (height, width) = imgX.shape
    
    img1 = numpy.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if imgX[i][j] != 0:
                angle = numpy.arctan(imgY[i][j] / imgX[i][j])
            elif imgX[i][j] == 0 and imgY[i][j] != 0:
                angle = numpy.pi / 2
            else:
                continue
            #    0    : [0, 1]
            #    45   : [1, 1]
            #    90   : [1, 0]
            #    135  : [-1, 1]
            # 0 or 180
            if angle < 0.39269908169872414 and angle > -0.39269908169872414:
                img1[i][j] = _compare(img[i - 1: i + 1, j - 1: j + 1], [1, 0])
            
            # 45 or 225
            if angle < 1.1780972450961724 and angle > 0.39269908169872414:
                img1[i][j] = _compare(img[i - 1: i + 1, j - 1: j + 1], [1, 1])
            
            # 90 or 270
            if numpy.abs(angle) < 1.5707963267948966 and numpy.abs(angle) > 1.1780972450961724:
                img1[i][j] = _compare(img[i - 1: i + 1, j - 1: j + 1], [0, 1])
            
            # 135 or 315
            if angle < -0.7853981633974484 and angle > -1.1780972450961724:
                img1[i][j] = _compare(img[i - 1: i + 1, j - 1: j + 1], [-1, 1])
            
    return img1

def _compare(img, direction):
    value = 0
    # test the value with other 2 values around it
    # handle outbound error
    try:
        # if smaller than one of others, return 0
        if img[1, 1] < img[1 + direction[0], 1 + direction[1]]:
            return 0
        else:
            value = img[1, 1]
    except:
        pass
    try:
        # same with up
        if img[1, 1] < img[1 - direction[0], 1 - direction[1]]:
            return 0
        else:
            value = img[1, 1]
    except:
        pass
    # otherwise, use original value
    return value
    
def _convolution(img, kernel):
    (height, width) = img.shape

    (kerHeight, kerWidth) = kernel.shape
    
    # fill image edge with 0
    new1 = numpy.zeros((height, int(kerWidth / 2)))
    new2 = numpy.zeros((int(kerHeight / 2), width + new1.shape[1] * 2))
    
    img1 = numpy.hstack([new1, numpy.hstack([img, new1])])
    img1 = numpy.vstack([new2, numpy.vstack([img1, new2])])
    
    '''
    # fill image edge with 0
    img1 = numpy.zeros((height + kerHeight - 1, width + kerWidth - 1))
    
    for i in range(height):
        for j in range(width):
            img1[i + int(kerHeight / 2)][j + int(kerWidth / 2)] = img[i][j]
    '''
    
    # create output image
    conv = numpy.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            conv[i][j] = elementSum(img1[i:i + kerHeight,j:j + kerWidth ],kernel)

    return conv
    
# calculate element sum by kernel
def elementSum(img, kernel):
    res = (img * kernel).sum()
    if(res < 0):
        res = 0
    elif res > 255:
        res  = 255
    return res