import os
from sys import platform
import numpy as np
import cv2 as cv
import time
from PIL import Image
import json
import datetime

clear_terminal = None

COLOR = (255,0,0)
STROKE = 2

HIGHEST_RESOLUTION = (1024, 768)
CURRENT_RESOLUTION = (320, 240)
ZOOM = True

face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')

def clear_windows():
    os.system('cls')

def clear_linux():
    os.system('clear')

if platform == 'linux' or platform == 'linux2':
    clear_terminal = clear_linux
elif platform == 'darwin':
    clear_terminal = clear_linux
elif platform == 'win32':
    clear_terminal = clear_windows
    os.system('@echo off')
    clear_terminal()

if clear_terminal == None:
    print('Végzetes hiba lépett fel, ezért a program fel lett függesztve.')
    exit()


print('Válasszon kamera felbontást:\n1 | 320 x 240\n2 | 640 x 480\n3 | 800 x 600\n4 | 1024 x 768')
resolution = int(input('>>>'))
if resolution == 1:
    CURRENT_RESOLUTION = (320, 240)
    ZOOM = True
elif resolution == 2:
    CURRENT_RESOLUTION = (640, 480)
    ZOOM = False
elif resolution == 3:
    CURRENT_RESOLUTION = (800, 600)
    ZOOM = False
elif resolution == 4:
    CURRENT_RESOLUTION = (1024, 768)
    ZOOM = False
    
def train():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "registeredMembers")

    recognizer = cv.face_LBPHFaceRecognizer.create()

    current_id = 0
    label_ids = {}
    labels = []
    train = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                #print(label, path)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1

                id_ = label_ids[label]
                pil_image = Image.open(path).convert('L')
                image_array = np.array(pil_image, 'uint8')
                faces = face_cascade.detectMultiScale(image_array, 1.2, 4)

                for (x,y,w,h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    train.append(roi)
                    labels.append(id_)

    recognizer.train(train, np.array(labels))
    recognizer.save('trainer.yml')

def registerMemberPicture(data):
    CAP = cv.VideoCapture(0)
    #CAP.set(3, 1280)
    #CAP.set(4, 720)

    memberslist = []
    with open('members.json', 'r', encoding='UTF-8') as members:
        memberslist = json.load(members)
    nextMemberId = len(memberslist['members'])

    while True:
        ret, frame = CAP.read()
        if not ret: 
            print('Hiba lépett fel a kamera indítása közben.')
            break
        frameGreyscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frameGreyscale, 1.2, 4)

        if cv.waitKey(20) & 0xFF == ord('q'):
            if len(faces) > 0:
                os.mkdir(f'registeredMembers/{nextMemberId}')
                for i in range(3):
                    for (x,y,w,h) in faces:
                        cv.imwrite(f'registeredMembers/{nextMemberId}/{i}.png', frame[y:y+h, x:x+w])
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



        #print(faces, len(faces))

        for (x,y,w,h) in faces:
            cv.rectangle(frame, (x,y), (x+w,y+h), COLOR, STROKE )

        #frame = cv.resize(frame, (CURRENT_RESOLUTION[0]*2, CURRENT_RESOLUTION[1]*2))

        cv.imshow('frame', frame)
    CAP.release()
    cv.destroyAllWindows()
    train()
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

def recognize():
    memberslist = []
    with open('members.json', 'r', encoding='UTF-8') as members:
        memberslist = json.load(members)
    print(memberslist)
    if not len(memberslist['members']) == 0:
        # date = datetime.datetime.now()
        # print(date)
        recognizer = cv.face_LBPHFaceRecognizer.create()
        recognizer.read('trainer.yml')
        CAP = cv.VideoCapture(0)
        CAP.set(3, CURRENT_RESOLUTION[0])
        CAP.set(4, CURRENT_RESOLUTION[1])
        while True:
            ret, frame = CAP.read()

            if not ret: 
                print('Hiba lépett fel a kamera indítása közben.')
                break

            if cv.waitKey(20) & 0xFF == ord('q'):
                break

            frameGreyscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frameGreyscale, 1.2, 4)

            for (x,y,w,h) in faces:
                roi_gray = frameGreyscale[y:y+h, x:x+h]
                id_, cf = recognizer.predict(roi_gray)
                print(memberslist['members'][id_], cf)

            #print(faces, len(faces))

            for (x,y,w,h) in faces:
                cv.rectangle(frame, (x,y), (x+w,y+h), COLOR, STROKE )

<<<<<<< HEAD
            #frame = cv.resize(frame, (CURRENT_RESOLUTION[0]*2, CURRENT_RESOLUTION[1]*2))
=======
        frame = cv.resize(frame, (CURRENT_RESOLUTION[0]*2, CURRENT_RESOLUTION[1]*2))
>>>>>>> 34a87274e95b8cb1fb66b87c2d17d3368a98b1ee

            cv.imshow('frame', frame)
        CAP.release()
        cv.destroyAllWindows()
        os.system('cls')
        main()
    else:
        os

def main():
    print("Beszélő Benedek Fitness CLI -\n© Hajdu Benedek, Resz Máté, Granilla Péter\n")

    print("1 | Arcfelismerés")
    print("2 | Új tagság regisztrálása")
    print("x | Kilépés")
    choice = input(">>> ")

    if choice == "1":
        recognize()
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