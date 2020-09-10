import imutils, cv2, numpy as np

OPENCV_OBJECT_TRACKERS = {
	'csrt': cv2.TrackerCSRT_create,
	'kcf': cv2.TrackerKCF_create,
	'boosting': cv2.TrackerBoosting_create
        }
		
video = cv2.VideoCapture('D:/Datasets/NIT Datasets/cutvideo.mp4')
trackers = cv2.MultiTracker_create()

while True:
    frame = video.read()[1]
    if frame is None:
        break
    frame = imutils.resize(frame, width=1500)
    success, boxes = trackers.update(frame)
    for box in boxes:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('s'):
        box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        if box != (0, 0, 0, 0):
            tracker = OPENCV_OBJECT_TRACKERS['kcf']()
            trackers.add(tracker, frame, box)
    elif key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
    
	
