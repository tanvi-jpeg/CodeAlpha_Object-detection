import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture('traffic.mp4') #use filename for importing videos

model = YOLO("yolov8n.pt")

while True:
        ret , frame = cap.read()
        results = model(frame)
        #results = model(frame,classes=[0]) - for only people
        annotated_frame = results[0].plot()
        cv2.imshow("Live Camera Feed", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()