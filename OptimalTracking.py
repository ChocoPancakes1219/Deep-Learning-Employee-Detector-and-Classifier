import cv2
from ultralytics import YOLO
from numpy import Inf
import math

# Import model
model=YOLO("best_optimized.pt")

# Load video input and employee id list
cap=cv2.VideoCapture('sample.mp4')
employee_id=[]

while True:
    # Extract frame from video
    ret, frame = cap.read()

    if not ret:
        break

    # Get the boxes and track IDs
    results = model.track(frame, tracker="bytetrack.yaml")
    boxes = results[0].boxes.xyxy.cpu().tolist()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    track_cls = results[0].boxes.cls.int().cpu().tolist()

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Plot the tracks
    for box, cls in zip(boxes, track_cls):
        if cls==0:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            min_dist=Inf
            closest_id=""

            # Find the nearest person to the name tag
            for pbox, ptrack_id, pcls in zip(boxes, track_ids, track_cls):
                if pcls==1:
                    px1 = int(pbox[0])
                    py1 = int(pbox[1])
                    px2 = int(pbox[2])
                    py2 = int(pbox[3])
                    pcx = int(px1 + px2) // 2
                    pcy = int(py1 + py2) // 2
                    if min(min_dist,math.dist([cx,cy],[pcx,pcy]))!=closest_id:
                        closest_id=ptrack_id

            # add the id of the person to employee id list
            if closest_id not in employee_id:
                employee_id.append(closest_id)

    if employee_id:
        print(employee_id)
        for ebox, etrack_id, ecls in zip(boxes, track_ids, track_cls):
            if etrack_id in employee_id:
                ex1 = int(ebox[0])
                ey1 = int(ebox[1])
                ex2 = int(ebox[2])
                ey2 = int(ebox[3])
                ex = int(ex1 + ex2) // 2
                ey = int(ey1 + ey2) // 2
                cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (164, 87, 41), 3)
                cv2.putText(frame, "Employee ("+str(ex)+","+str(ey)+")", (ex1, ey1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 22, 12), 1)

    cv2.imshow("Display", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()