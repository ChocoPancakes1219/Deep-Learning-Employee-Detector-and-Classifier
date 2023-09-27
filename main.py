import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

cap=cv2.VideoCapture("sample.mp4")

first_frame=None

while True:
    ret, frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)
    if first_frame is None:
        first_frame=gray
        continue
    delta_frame=cv2.absdiff(first_frame,gray)
    threshold_frame=cv2.threshold(delta_frame,50,255,cv2.THRESH_BINARY)[1]
    threshold_frame = cv2.dilate(threshold_frame,None,iterations=2)
    (contours, _) = cv2.findContours(threshold_frame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area<1000:
            continue
        x,y,w,h=cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    cv2.imshow("Frame",frame)

    key=cv2.waitKey(1)
    if key==ord('q'):
        break




cap.release()
cv2.destroyAllWindows()

