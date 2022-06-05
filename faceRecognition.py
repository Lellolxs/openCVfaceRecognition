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
resolution = int(input('>>> '))
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
clear_terminal()

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
    CAP = cv.VideoCapture(0, cv.CAP_DSHOW)
    CAP.set(3, CURRENT_RESOLUTION[0])
    CAP.set(4, CURRENT_RESOLUTION[1])

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
            
        for (x,y,w,h) in faces:
            cv.rectangle(frame, (x,y), (x+w,y+h), COLOR, STROKE )

        #frame = cv.resize(frame, (CURRENT_RESOLUTION[0]*2, CURRENT_RESOLUTION[1]*2))

        cv.imshow('frame', frame)
    CAP.release()
    cv.destroyAllWindows()
    train()
    clear_terminal()
    main()

def getBirthDate():
    szuletett = input("Született (ÉV-HÓ-NAP): ")
    szuletett = szuletett.split('-')
    if len(szuletett) == 3:
        for i, v in enumerate(szuletett):
            szuletett[i] = int(v)
        return szuletett
    else:
        getBirthDate()

def registerData():
    date = datetime.datetime.now()
    userData = list()
    clear_terminal()
    userData.append(input("Név: "))
    userData.append(getBirthDate())
    userData.append([date.year, date.month, date.day])
    userData.append(int(input("Bérlet típus (ÉRVÉNYESSÉG): ")))
    clear_terminal()
    registerMemberPicture(userData)

def recognize():
    memberslist = []
    with open('members.json', 'r', encoding='UTF-8') as members:
        memberslist = json.load(members)
    if not len(memberslist['members']) == 0:
        recognizedFaces = dict()
        date = datetime.datetime.now()
        recognizer = cv.face_LBPHFaceRecognizer.create()
        recognizer.read('trainer.yml')
        CAP = cv.VideoCapture(0, cv.CAP_DSHOW)
        CAP.set(3, CURRENT_RESOLUTION[0])
        CAP.set(4, CURRENT_RESOLUTION[1])

        frames = 0
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
                cv.rectangle(frame, (x,y), (x+w,y+h), COLOR, STROKE )
                if not id_ in recognizedFaces:
                    recognizedFaces[id_] = [id_, 0]
                else:
                    recognizedFaces[id_][1] += 1
                frames += 1

            if ZOOM:
                frame = cv.resize(frame, (CURRENT_RESOLUTION[0]*2, CURRENT_RESOLUTION[1]*2))
            cv.imshow('frame', frame)
            if frames > 25:
                CAP.release()
                cv.destroyAllWindows()
                for k, v in recognizedFaces.items():
                    id = recognizedFaces[k][0]
                    percentage = recognizedFaces[k][1]
                    print(memberslist['members'][id]['name'], '-', str(percentage*4)+'%')
                    input('Nyomj enter-t a továbblépéshez.\n')
                break

        clear_terminal()
        main()
    else:
        clear_terminal()
        main()

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
        clear_terminal()
        exit()
    else: 
        print("Helytelen érték.")
        clear_terminal()
        main()
main()