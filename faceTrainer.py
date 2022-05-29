import os
from PIL import Image
import numpy as np
import cv2 as cv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "registeredMembers")
face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')

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