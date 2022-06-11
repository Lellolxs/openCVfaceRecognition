import os
import shutil
from sys import platform
import numpy as np
import cv2 as cv
import time
from PIL import ImageFont, ImageDraw, Image
import json
import datetime

clear_terminal = None

COLOR = (255,0,0)
STROKE = 2

HIGHEST_RESOLUTION = (1024, 768)
CURRENT_RESOLUTION = (320, 240)
ZOOM = True
FONT = cv.FONT_HERSHEY_COMPLEX
face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
MPLUS_FONT = ImageFont.truetype('./redist/MPLUS.ttf', 32)

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
                faces = face_cascade.detectMultiScale(image_array, 1.15, 4)

                if len(faces) == 1:
                    for (x,y,w,h) in faces:
                        roi = image_array[y:y+h, x:x+w]
                        train.append(roi)
                        labels.append(id_)
                else:
                    print("Meghibásodott fényképekkel nem lehet tanítani.\nValószínüleg gyenge fényviszonyok, vagy rosz kamera elhelyezés okozhatták ezt a hibát.")
                    input('Nyomj enter-t a továbblépéshez.\n')
                    return False

    recognizer.train(train, np.array(labels))
    recognizer.save('trainer.yml')
    return True


def registerMemberPicture(data):
    CAP = cv.VideoCapture(0, cv.CAP_DSHOW)
    CAP.set(cv.CAP_PROP_FRAME_WIDTH, CURRENT_RESOLUTION[0])
    CAP.set(cv.CAP_PROP_FRAME_HEIGHT, CURRENT_RESOLUTION[1])
    # CAP.set(cv.CAP_PROP_SETTINGS, 1)
    # print("Kérjük konfigurálja a kamerát.\n")
    # input('Nyomjon entert a folytatáshoz:\n')
    
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
        faces = face_cascade.detectMultiScale(frameGreyscale, 1.15, 4)

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

        cv.imshow('Arc Beolvasasa', frame)
    CAP.release()
    cv.destroyAllWindows()
    trained = train()
    if not trained:
        members = memberslist['members']
        memberslist['members'].pop(nextMemberId)
        shutil.rmtree( f'./registeredMembers/{nextMemberId}')
        with open('members.json', 'w', encoding='UTF-8') as members:
            json.dump(memberslist, members, indent=2, separators=(',',': '), ensure_ascii=False)
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
    clear_terminal()
    memberslist = []
    with open('members.json', 'r', encoding='UTF-8') as members:
        memberslist = json.load(members)
    if not len(memberslist['members']) == 0: 
        recognizedFaces = dict()
        recognizer = cv.face_LBPHFaceRecognizer.create()
        recognizer.read('trainer.yml')
        CAP = cv.VideoCapture(0, cv.CAP_DSHOW)
        # CAP.set(cv.CAP_PROP_SETTINGS, 1)
        # print("Kérjük konfigurálja a kamerát.\n")
        # input('Nyomjon entert a folytatáshoz:\n')
        CAP.set(cv.CAP_PROP_FRAME_WIDTH, CURRENT_RESOLUTION[0])
        CAP.set(cv.CAP_PROP_FRAME_HEIGHT, CURRENT_RESOLUTION[1])
        mostLikelyMatch = ''
        frames = 0
        while True:
            cv.waitKey(1)
            ret, frame = CAP.read()
            if not ret: 
                print('Hiba lépett fel a kamera indítása közben.')
                break

            frameGreyscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frameGreyscale, 1.15, 4)

            if len(faces) > 0 and len(faces) < 2:
                for (x,y,w,h) in faces:
                    roi_gray = frameGreyscale[y:y+h, x:x+h]
                    id_, cf = recognizer.predict(roi_gray)
                    cv.rectangle(frame, (x,y), (x+w,y+h), COLOR, STROKE )

                    print(cf)
                    if cf < 50 and not id_ in recognizedFaces:
                        recognizedFaces[id_] = [id_, 1]
                    elif cf < 50 and id_ in recognizedFaces:
                        recognizedFaces[id_][1] += 1
                    elif cf > 50:
                        if not 'unknown' in recognizedFaces:
                            recognizedFaces['unknown'] = ['unknown', 1]
                        else:
                            recognizedFaces['unknown'][1] += 1
                    frames += 1

                highestPercentage = 0
                highestKey = ''
                for k, v in recognizedFaces.items():
                    if v[1] > highestPercentage:
                        highestKey = k
                        highestPercentage = v[1]
                        
                id = recognizedFaces[highestKey][0]
                
                if id == 'unknown':
                    mostLikelyMatch = 'Legjobb találat: Ismeretlen!'
                else:
                    memeberdata = memberslist['members'][id]
                    mostLikelyMatch = 'Legjobb találat: %s' % memeberdata['name']
                    
            frame_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(frame_pil)
            draw.text((10, CURRENT_RESOLUTION[1]-60), mostLikelyMatch, font = MPLUS_FONT, fill = (0, 0, 250, 0))
            frame = np.array(frame_pil)
            progress = f'Arcfelismeres folyamatban: {frames*2}%'
            cv.putText(frame,progress,(10,30), FONT ,1,(25,25,250),1)

            if ZOOM:
                frame = cv.resize(frame, (CURRENT_RESOLUTION[0]*2, CURRENT_RESOLUTION[1]*2))
            cv.imshow('Arcfelismerés', frame)
            if frames > 50:
                CAP.release()
                cv.destroyAllWindows()
                date = datetime.date.today()
                # highestPercentage = 0
                # highestKey = ""

                # for k, v in recognizedFaces.items():
                #     # print("value", v)
                #     if v[1] > highestPercentage:
                #         highestKey = k
                #         highestPercentage = v[1]

                id = recognizedFaces[highestKey][0]
                if id != 'unknown':
                    memeberdata = memberslist['members'][id]
                    birthdate = memeberdata['birthdate']
                    purchase_date = datetime.date(memeberdata['purchase_date'][0], memeberdata['purchase_date'][1], memeberdata['purchase_date'][2])
                    pass_expires = purchase_date + datetime.timedelta(memeberdata['pass_type'])
                    delta = pass_expires - date
                    percentage = recognizedFaces[highestKey][1]
                    clear_terminal()

                    print("Név: ", memeberdata['name'], '-', str(percentage*2)+'%')
                    print("Született: ", f"{birthdate[0]} {birthdate[1]} {birthdate[2]}.")
                    print("Bérlet típus: ", memeberdata['pass_type'])
                    print("Bérletet vásárolta: ", f"{purchase_date.year} {purchase_date.month} {purchase_date.day}.")

                    if not delta.days == 0:
                        print("Bérlete lejár : ", f"{pass_expires.year} {pass_expires.month} {pass_expires.day}. ({delta.days} Nap)\n")
                    else:
                        print("Bérlete lejár : ", f"{pass_expires.year} {pass_expires.month} {pass_expires.day}. ( Lejárt )\n")

                    input('Nyomj enter-t a továbblépéshez.\n')
                    break
                else:
                    clear_terminal()
                    print('Ismeretlen személy.\n')
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
    # print("3 | Tagság megvonása")
    print("x | Kilépés")
    choice = input(">>> ")

    if choice == "1":
        recognize()
    elif choice == "2":
        registerData()
    # elif choice == "3":
        #revokeMembership()
    elif choice == "x":
        clear_terminal()
        exit()
    else: 
        print("Helytelen érték.")
        clear_terminal()
        main()
main()