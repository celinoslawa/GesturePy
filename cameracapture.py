import cv2
import numpy as np
import hog
import imProc
import SVM
import tkinter as tk
from PIL import Image, ImageTk
#np.set_printoptions(threshold=np.nan)

##################TRAINING
hogD = hog.Hog()
hog_descriptors = hogD.getHOG()

#print("Descriptor: ", hog_descriptors)
print("Descriptor length: ", hog_descriptors.shape)
responses = hogD.getResp()

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

#Button window
buttonFrame = tk.Frame(window, width=300, height=100)
buttonFrame.grid(row = 300, column=0, padx=10, pady=2)
appStatus = tk.IntVar()
C1 = tk.Checkbutton(buttonFrame, text = "Calibrated", variable = appStatus, \
                 onvalue = 1, offvalue = 0, height=5, \
                 width = 20)
C1.place(x=10, y=10)
C1.pack(side =tk.LEFT)
#Text window
#textFrame = tk.Frame(window, width=300, height=100)
#textFrame.grid(row = 300, column=0, padx=10, pady=2)
T1 = tk.Text(buttonFrame, height=5, width = 30, bg = "#e5e5e5", font=("Helvetica", 17))


###################################### CAMERA CAPTURE
#Capture video frames


lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)
cap = cv2.VideoCapture(1)

if cap.isOpened() == False:
    print ("VideoCapture failed")

iP = imProc.ImProc()

def show_frame():
    T1.delete('1.0', tk.END)
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    mask1 = cv2.resize(frame, (600, 800), interpolation=cv2.INTER_AREA)
    mask2 = mask1[0:800, 0:480]
    mask2 = cv2.GaussianBlur(mask2, (5, 5), 0)
    #mHSV = cv2.cvtColor(mask2, cv2.COLOR_BGR2HSV)
    mask3 = iP.backgroungRemove(mask2, appStatus.get())
    #mask4 = iP.backgroungRemove1(mask2, appStatus)
    print("appStatus = ", appStatus.get())
    if appStatus.get() == 0:
        mask2 = iP.drawCalibrationPoints(mask2)
        T1.insert(tk.INSERT, "CALIBRATION ONGOING ...")
    else:
        descriptors = hogD.compute(mask3)
        descriptors = descriptors.T
        print("Descriptors Len: ", descriptors.shape)
        resp = model.predict(descriptors)
        T1.insert(tk.INSERT, resp)
        print("Predicted value :  ", resp)
    mask2 = iP.drawContours(mask2, mask3)
    mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(mask2)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(5, show_frame)
    T1.pack(side = tk.RIGHT)

show_frame()
window.mainloop()  #Starts GUI

