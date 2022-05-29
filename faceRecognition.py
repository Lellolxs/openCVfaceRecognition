import os
import numpy as np
import cv2 as cv
import time
COLOR = (255,0,0)
STROKE = 2
LAPTOP_RES = (640, 480) # idk, huawei laptop camera
DEFAULT_RES = (320, 240) # Logitech c110

CURRENT_RESOLUTION = LAPTOP_RES

face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
CAP = cv.VideoCapture(0)
CAP.set(3, CURRENT_RESOLUTION[0])
CAP.set(4, CURRENT_RESOLUTION[1])

def registerMember():
    conf = open('config.txt', 'r', encoding='UTF-8')
    PERSON_INDEX = int(conf.readline())
    conf.close()
    while True:
        ret, frame = CAP.read()

        if cv.waitKey(20) & 0xFF == ord('q'):
            if len(faces) > 0:
                os.mkdir(f'registeredMembers/{PERSON_INDEX}')
                for i in range(3):
                    print(PERSON_INDEX)                    
                    cv.imwrite(f'registeredMembers/{PERSON_INDEX}/{i}.png', frame)
                    time.sleep(1)
                PERSON_INDEX += 1
                break

        frameGreyscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frameGreyscale, 1.2, 4)

        #print(faces, len(faces))

        for (x,y,w,h) in faces:
            cv.rectangle(frame, (x,y), (x+w,y+h), COLOR, STROKE )

        #frame = cv.resize(frame, (CURRENT_RESOLUTION[0]*2, CURRENT_RESOLUTION[1]*2))

        cv.imshow('frame', frame)
    conf = open('config.txt', 'w', encoding='UTF-8')
    print(PERSON_INDEX, file=conf)
    CAP.release()
    cv.destroyAllWindows()

registerMember()