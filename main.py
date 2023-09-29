import cv2
from ultralytics import YOLO
import pandas as pd
from tracker import *
import numpy as np

import time

model=YOLO("best.pt")
object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)
tracker=EuclideanDistTracker()


cap=cv2.VideoCapture('sample.mp4')
employee_id=[]

while True:
    # Extract frame from video
    ret, frame = cap.read()


    mask=object_detector.apply(frame)
    contours, _ =cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detections=[]

    if not ret:
        break

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # object tracking
    boxes_ids = tracker.update(detections)



    results = model.predict(frame)
    # Get coordinates of name tags detected within the frame
    a = results[0].boxes.xyxy
    px = pd.DataFrame(a.cpu()).astype("float")

    for index, row in px.iterrows():

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(str(cx) + "," + str(cy)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 22, 12), 1)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            if x <= x1 and y <= y1 and (x + w) >= x2 and (y + h) >= y2:
                print(id)
                employee_id.append(id)


    if employee_id:
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            if id in employee_id:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)




    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
