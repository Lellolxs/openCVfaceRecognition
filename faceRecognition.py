from tkinter import Frame
import numpy as np
import cv2 as cv
COLOR = (255,0,0)
STROKE = 2

face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
CAP = cv.VideoCapture(0)
CAP.set(3, 320)
CAP.set(4, 240)

def registerMember():
    while True:
        ret, frame = CAP.read()

        if cv.waitKey(20) & 0xFF == ord('q'):
            if len(faces) > 0:
                for i in range(5):
                    cv.imwrite(f'registeredMembers/{i}.png', frame)
                break

        frameGreyscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frameGreyscale, 1.2, 4)

        print(faces, len(faces))

        for (x,y,w,h) in faces:
            cv.rectangle(frame, (x,y), (x+w,y+h), COLOR, STROKE )

        resized = cv.resize(frame, (640, 480))

        cv.imshow('frame', resized)
    CAP.release()
    cv.destroyAllWindows()

registerMember()