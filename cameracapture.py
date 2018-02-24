import cv2
import hog
import time
import imProc
import SVM
import tkinter as tk
from PIL import Image, ImageTk
#import Image, ImageTk

# 0 calibration 1 detectin
appStatus = 0

#################TRAINING

hog_descriptors = hog.getHOG()
responses = hog.getResp()

print('Training SVM model ...')
model = SVM.SVM()
model.train(hog_descriptors, responses)

####################Set up GUI

window = tk.Tk()  #Makes main window
window.wm_title("Hand Gesture Detection")
window.config(background="#FFFFFF")

#Graphics window
imageFrame = tk.Frame(window, width=800, height=480)
imageFrame.grid(row=0, column=0, padx=10, pady=2)

###################################### CAMERA CAPTURE
#Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)
cap = cv2.VideoCapture(1)

if cap.isOpened() == False:
    print ("VideoCapture failed")

iP = imProc.ImProc()

def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    mask1 = cv2.resize(frame, (600, 800), interpolation=cv2.INTER_AREA)
    mask2 = mask1[0:800, 0:480]
    mask2 = cv2.GaussianBlur(mask2, (5, 5), 0)
    mHSV = cv2.cvtColor(mask2, cv2.COLOR_BGR2HSV)
    mask3 = iP.backgroungRemove(mask2, appStatus)
    #mask4 = iP.backgroungRemove1(mask2, appStatus)
    if appStatus == 0:
        mask2 = iP.drawCalibrationPoints(mask2)
    mask2 = iP.drawContours(mask2, mask3)
    img = Image.fromarray(mask2)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

#Slider window (slider controls stage position)
sliderFrame = tk.Frame(window, width=600, height=100)
sliderFrame.grid(row = 600, column=0, padx=10, pady=2)


show_frame()  #Display 2
window.mainloop()  #Starts GUI


#cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,800)
#iP = imProc.ImProc()
#print('Camera Capturing  ... ')
#while(True):
#    ret, frame = cap.read()
#    if ret == False:
#        print("Frame is empty")
#
#    #print("height and width : ", frame.shape)
#    mask1 = cv2.resize(frame, (600, 800), interpolation=cv2.INTER_AREA)
#    mask2 = mask1[0:800, 0:480]
#    mask2 = cv2.GaussianBlur(mask2, (5, 5), 0)
#    mHSV = cv2.cvtColor(mask2, cv2.COLOR_BGR2HSV)
#    #mHSV = cv2.fastNlMeansDenoising(mHSV, None, 10, 7, 21)
#    mask3 = iP.backgroungRemove(mask2, appStatus)
#    #mask4 = iP.backgroungRemove1(mask2, appStatus)
#    if appStatus == 0:
#        mask2 = iP.drawCalibrationPoints(mask2)
#    mask2 = iP.drawContours(mask2, mask3)
#
#    # Display the resulting frame
#    cv2.imshow('frame', mask2)
#    cv2.imshow('frame1', mask3)
#    cv2.imshow('frame2', mHSV)
#    #cv2.imshow('frame3  emn', mask4)
#    #mask2
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#      break
#
## When everything done, release the capture
#cap.release()
#cv2.destroyAllWindows()