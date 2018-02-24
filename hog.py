import numpy as np
import cv2

#size of image
winSize = (60, 100)
blockSize = (40, 40)
#determines the overlap between neighboring blocks and controls the degree of contrast normalization
blockStride = (20, 20)
#The cellSize is chosen based on the scale of the features important to do the classification.
cellSize = (10, 10)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
#00-180 degrees with sign
signedGradient = True
N = 125

def getHOG():
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)
    hog_descriptors = []
    for i in range(0,10):
        gest = i
        print('GEST : %d' %gest)
        for j in range(1, N):
            img = cv2.imread('%d' %gest + 'mask/dys_%d.jpg' %j, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(img, (60, 100), interpolation=cv2.INTER_AREA)
            hog_descriptors.append(hog.compute(mask))
    #Remove single-dimensional entries from the shape of an array.
    hog_descriptors = np.squeeze(hog_descriptors)
    #responses = np.int32(np.repeat(np.arange(10),N-1)[:,np.newaxis])
    return hog_descriptors

def getResp():
    responses = np.int32(np.repeat(np.arange(10),N-1)[:,np.newaxis])
    return responses


