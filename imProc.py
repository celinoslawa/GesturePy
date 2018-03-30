import cv2
import numpy as np


class ImProc:
    # wybranie 7 pkt które znajdją sie na ręce
    #           a
    #       o       p
    #       f       g
    #      b    c    d
    #       m       n
    #           e
    def __init__(self):
        self.aX = 240
        self.aY = 50
        self.bX = 170
        self.bY = 480
        self.cX = 240
        self.cY = 480
        self.dX = 310
        self.dY = 480
        self.eX = 240
        self.eY = 700
        self.fX = 200
        self.fY = 350
        self.gX = 280
        self.gY = 350
        self.mX = 190
        self.mY = 580
        self.nX = 290
        self.nY = 580
        self.oX = 200
        self.oY = 200
        self.pX = 280
        self.pY = 200

        self.avUpperT = np.array([0, 0, 0])
        self.avUpperTmn = np.array([0, 0, 0])
        self.avUpperTAFG = np.array([0, 0, 0])
        self.avUpperTOP = np.array([0, 0, 0])
        self.avLowerT = np.array([0, 0, 0])
        self.avLowerTmn = np.array([0, 0, 0])
        self.avLowerTAFG = np.array([0, 0, 0])
        self.avLowerTOP = np.array([0, 0, 0])
        self.kernel = np.ones((15, 15), np.uint8)
        self.pointColor = [255, 0, 0]
        self.corr = 30


    def backgroungRemove(self, maskn, status):
        global a, b, c, d, f, g, e, m, n, o, p
        mHSV = cv2.cvtColor(maskn, cv2.COLOR_BGR2HSV)
        mHSV = cv2.medianBlur(mHSV, 5)
        cv2.blur(mHSV, (5, 5))
        cv2.blur(mHSV, (5, 5))
        cv2.dilate(mHSV, self.kernel, 1)
        cv2.erode(mHSV, self.kernel, 3)
        a = mHSV[self.aY, self.aX]
        b = mHSV[self.bY, self.bX]
        c = mHSV[self.cY, self.cX]
        d = mHSV[self.dY, self.dX]
        e = mHSV[self.eY, self.eX]
        f = mHSV[self.fY, self.fX]
        g = mHSV[self.gY, self.gX]
        m = mHSV[self.mY, self.mX]
        n = mHSV[self.nY, self.nX]
        o = mHSV[self.oY, self.oX]
        p = mHSV[self.pY, self.pX]
        if (status == 0):
            self.calibrationOfTreshold()
            self.calibrationOfTresholdEMN()
            self.calibrationOfTresholdAFG()
            self.calibrationOfTresholdOP()
        mTresh = cv2.inRange(mHSV, self.avLowerT, self.avUpperT)
        mTresh2 = cv2.inRange(mHSV, self.avLowerTmn, self.avUpperTmn)
        mTresh3 = cv2.inRange(mHSV, self.avLowerTAFG, self.avUpperTAFG)
        mTresh4 = cv2.inRange(mHSV, self.avLowerTOP, self.avUpperTOP)
        maskedImg = cv2.bitwise_or(mTresh, mTresh2)
        maskedImg = cv2.bitwise_or(maskedImg, mTresh3)
        maskedImg = cv2.bitwise_or(maskedImg, mTresh4)
        print("avLowerT = ", self.avLowerT, self.avLowerTmn)
        print("avUpperT = ", self.avUpperT, self.avUpperTmn)
        cv2.blur(maskedImg, (10,10))
        cv2.dilate(maskedImg, self.kernel, 1)
        cv2.erode(maskedImg, self.kernel, 3)
        return maskedImg

    # def backgroungRemove1(self, maskn, status):
    #     global e, m, n
    #     mHSV = cv2.cvtColor(maskn, cv2.COLOR_BGR2HSV)
    #     mHSV = cv2.medianBlur(mHSV, 5)
    #     cv2.blur(mHSV, (5, 5))
    #     cv2.dilate(mHSV, self.kernel, 1)
    #     cv2.erode(mHSV, self.kernel, 3)
    #     # cv2.split(mHSV)
    #     e = mHSV[self.eY, self.eX]
    #     m = mHSV[self.mY, self.mX]
    #     n = mHSV[self.nY, self.nX]
    #
    #     if (status == 0):
    #         self.calibrationOfTreshold1()
    #     mTresh2 = cv2.inRange(mHSV, self.avLowerTmn, self.avUpperTmn)
    #     return mTresh2

    def drawContours( self, frame, maskn):
        im2, contours, hierarchy = cv2.findContours(maskn,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)
        #mRgba, contours, hierarhy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return frame

    def drawCalibrationPoints(self, maskn):
        cv2.rectangle(maskn, (self.aX - 5, self.aY - 5), (self.aX + 5, self.aY + 5), (0, 255, 0))
        cv2.rectangle(maskn, (self.bX - 5, self.bY - 5), (self.bX + 5, self.bY + 5), (0, 255, 0))
        cv2.rectangle(maskn, (self.cX - 5, self.cY - 5), (self.cX + 5, self.cY + 5), (0, 255, 0))
        cv2.rectangle(maskn, (self.dX - 5, self.dY - 5), (self.dX + 5, self.dY + 5), (0, 255, 0))
        cv2.rectangle(maskn, (self.eX - 5, self.eY - 5), (self.eX + 5, self.eY + 5), (0, 255, 0))
        cv2.rectangle(maskn, (self.fX - 5, self.fY - 5), (self.fX + 5, self.fY + 5), (0, 255, 0))
        cv2.rectangle(maskn, (self.gX - 5, self.gY - 5), (self.gX + 5, self.gY + 5), (0, 255, 0))
        cv2.rectangle(maskn, (self.mX - 5, self.mY - 5), (self.mX + 5, self.mY + 5), (0, 255, 0))
        cv2.rectangle(maskn, (self.nX - 5, self.nY - 5), (self.nX + 5, self.nY + 5), (0, 255, 0))
        cv2.rectangle(maskn, (self.oX - 5, self.oY - 5), (self.oX + 5, self.oY + 5), (0, 255, 0))
        cv2.rectangle(maskn, (self.pX - 5, self.pY - 5), (self.pX + 5, self.pY + 5), (0, 255, 0))
        maskn[self.aY, self.aX] = self.pointColor
        maskn[self.bY, self.bX] = self.pointColor
        maskn[self.cY, self.cX] = self.pointColor
        maskn[self.dY, self.dX] = self.pointColor
        maskn[self.eY, self.eX] = self.pointColor
        maskn[self.fY, self.fX] = self.pointColor
        maskn[self.gY, self.gX] = self.pointColor
        maskn[self.mY, self.mX] = self.pointColor
        maskn[self.nY, self.nX] = self.pointColor
        maskn[self.oY, self.oX] = self.pointColor
        maskn[self.pY, self.pX] = self.pointColor
        return maskn

    def calibrationOfTreshold(self):
        values = []
        upper = [0, 0, 0]
        lower = [0, 0, 0]
        for i in range(0,3):
            #values.append(a[i])
            values.append(b[i])
            values.append(c[i])
            values.append(d[i])
            #values.append(f[i])
            #values.append(g[i])
            sorted(values)
            lower[i] = values[0]
            upper[i] = values[2] #6
            values.clear()
        self.avUpperT = np.asarray([upper[0] + 25, upper[1] + 25, 255])
        lower[0] = lower[0] - 30
        if (lower[0])<0:
            lower[0] = 0
        self.avLowerT = np.asarray([lower[0], lower[1] - 20, 0])

    def calibrationOfTresholdEMN(self):
        valuesmn = []
        uppermn = [0, 0, 0]
        lowermn = [0, 0, 0]
        for i in range(0, 3):
            valuesmn.append(e[i])
            valuesmn.append(m[i])
            valuesmn.append(n[i])
            sorted(valuesmn)
            lowermn[i] = valuesmn[0]
            uppermn[i] = valuesmn[2]
            valuesmn.clear()
        self.avUpperTmn = np.asarray([uppermn[0] + self.corr, uppermn[1] + self.corr, 255])

        lowermn[0] = lowermn[0] - 30
        if (lowermn[0]) < 0:
            lowermn[0] = 0
        self.avLowerTmn = np.asarray([lowermn[0], lowermn[1] - 20, 0])

    def calibrationOfTresholdAFG(self):
        valuesAFG = []
        upperAFG = [0, 0, 0]
        lowerAFG = [0, 0, 0]
        for i in range(0, 3):
            valuesAFG.append(a[i])
            valuesAFG.append(f[i])
            valuesAFG.append(g[i])
            sorted(valuesAFG)
            lowerAFG[i] = valuesAFG[0]
            upperAFG[i] = valuesAFG[2]
            valuesAFG.clear()
        self.avUpperTAFG = np.asarray([upperAFG[0] + self.corr, upperAFG[1] + self.corr, 255])
        lowerAFG[0] = lowerAFG[0] - 30
        if (lowerAFG[0]) < 0:
            lowerAFG[0] = 0
        self.avLowerTAFG = np.asarray([lowerAFG[0], lowerAFG[1] - 20, 0])

    def calibrationOfTresholdOP(self):
        valuesOP = []
        upperOP = [0, 0, 0]
        lowerOP = [0, 0, 0]
        for i in range(0, 3):
            valuesOP.append(o[i])
            valuesOP.append(p[i])
            sorted(valuesOP)
            lowerOP[i] = valuesOP[0]
            upperOP[i] = valuesOP[1]
            valuesOP.clear()
        self.avUpperOP = np.asarray([upperOP[0] + self.corr, upperOP[1] + self.corr, 255])
        lowerOP[0] = lowerOP[0] - 30
        if (lowerOP[0]) < 0:
            lowerOP[0] = 0
        self.avLowerTOP = np.asarray([lowerOP[0], lowerOP[1] - 20, 0])
