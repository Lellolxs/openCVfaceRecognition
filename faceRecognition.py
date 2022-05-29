import os
import numpy as np
import cv2 as cv
import time
from PIL import Image
import json

COLOR = (255,0,0)
STROKE = 2

HI_RES = (640, 480) # idk, huawei laptop camera
LOW_RES = (320, 240) # Logitech c110

CURRENT_RESOLUTION = LOW_RES

face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
recognizer = cv.face_LBPHFaceRecognizer.create()
recognizer.read('trainer.yml')

CAP = cv.VideoCapture(0)
CAP.set(3, CURRENT_RESOLUTION[0])
CAP.set(4, CURRENT_RESOLUTION[1])

def registerMemberPicture(data):
    memberslist = []
    with open('members.json', 'r', encoding='UTF-8') as members:
        memberslist = json.load(members)

    while True:
        ret, frame = CAP.read()
        if cv.waitKey(20) & 0xFF == ord('q'):
            if len(faces) > 0:
                os.mkdir(f'registeredMembers/{data[0]}')
                for i in range(3):                 
                    cv.imwrite(f'registeredMembers/{data[0]}/{i}.png', frame)
                    time.sleep(0.5)
                memberslist['members'].append({
                    "name": data[0],
                    "birthdate": data[1],
                    "purchase_date": data[2],
                    "pass_type": data[3]
                })
                with open('members.json', 'w', encoding='UTF-8') as members:
                    json.dump(memberslist, members, indent=2, separators=(',',': '), ensure_ascii=False)
                break

        frameGreyscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frameGreyscale, 1.2, 4)


        #print(faces, len(faces))

        for (x,y,w,h) in faces:
            roi_gray = frameGreyscale[y:y+h, x:x+h]
            id_, cf = recognizer.predict(roi_gray)
            print(id_, cf)
            cv.rectangle(frame, (x,y), (x+w,y+h), COLOR, STROKE )

        #frame = cv.resize(frame, (CURRENT_RESOLUTION[0]*2, CURRENT_RESOLUTION[1]*2))

        cv.imshow('frame', frame)
    CAP.release()
    cv.destroyAllWindows()
    os.system('cls')
    main()

def registerData():
    userData = list()
    os.system('cls')
    userData.append(input("Név: "))
    userData.append(input("Született (ÉV-HÓ-NAP): "))
    userData.append(input("Dátum: (ÉV-HÓ-NAP): "))
    userData.append(int(input("Bérlet típus (ÉRVÉNYESSÉG): ")))
    os.system('cls')
    registerMemberPicture(userData)

def main():
    print("Beszélő Benedek Fitness CLI -\n© Hajdu Benedek, Resz Máté, Granilla Péter\n")

    print("1 | Arcfelismerés")
    print("2 | Új tagság regisztrálása")
    print("x | Kilépés")
    choice = input(">>> ")

    if choice == "1":
        pass
    elif choice == "2":
        registerData()
    elif choice == "x":
        os.system('cls')
        exit()
    else: 
        print("Helytelen érték.")
        os.system('cls')
        main()
main()