import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')

cap = cv.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 480)

color = (255,0,0)
stroke = 3
while True:
    ret, frame = cap.read()
    frameGreyscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frameGreyscale, 1.2, 4)
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w,y+h), color, stroke )
    cv.imshow('frame', frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()