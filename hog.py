import numpy as np
import cv2

class Hog:
    def __init__(self):
        # size of image
        self.winSize = (60, 100)
        self.blockSize = (40, 40)
        #determines the overlap between neighboring blocks and controls the degree of contrast normalization
        self.blockStride = (20, 20)
        #The cellSize is chosen based on the scale of the features important to do the classification.
        self.cellSize = (10, 10)
        self.nbins = 9
        self.derivAperture = 1
        self.winSigma = -1.
        self.histogramNormType = 0
        self.L2HysThreshold = 0.2
        self.gammaCorrection = 1
        self.nlevels = 64
        #00-180 degrees with sign
        self.signedGradient = True
        self.N = 125
        #self.frameB
        self.hog = cv2.HOGDescriptor(self.winSize, self.blockSize, self.blockStride, self.cellSize, self.nbins, self.derivAperture, self.winSigma,
                                self.histogramNormType, self.L2HysThreshold, self.gammaCorrection, self.nlevels, self.signedGradient)

    def getHOG(self):
        #hog = cv2.HOGDescriptor(self.winSize, self.blockSize, self.blockStride, self.cellSize, self.nbins, self.derivAperture, self.winSigma,
        #                       self.histogramNormType, self.L2HysThreshold, self.gammaCorrection, self.nlevels, self.signedGradient)
        hog_descriptors = []
        for i in range(0,10):
            gest = i
            print('GEST : %d' %gest)
            for j in range(1, self.N):
                img = cv2.imread('%d' %gest + 'mask/dys_%d.jpg' %j, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(img, (60, 100), interpolation=cv2.INTER_AREA)
                #print(mask.shape)
                #print(mask.size)
                hogD  = self.hog.compute(mask)
                #print("HOG :" , len(hogD))
                hog_descriptors.append(self.hog.compute(mask))
        #Remove single-dimensional entries from the shape of an array.
        hog_descriptors = np.squeeze(hog_descriptors)
        #responses = np.int32(np.repeat(np.arange(10),N-1)[:,np.newaxis])
        return hog_descriptors

    def getResp(self):
        responses = np.int32(np.repeat(np.arange(10),self.N-1)[:,np.newaxis])
        return responses

    def compute(self, frame):
        #frameB = cv2.resize(frame, self.winSize,1,1, cv2.INTER_AREA)
        frame = cv2.resize(frame, (60, 100), interpolation=cv2.INTER_AREA)
        descriptors = self.hog.compute(frame)
        #descriptors = np.squeeze(descriptors)
        return descriptors



