#date: 2024-05-16T17:01:47Z
#url: https://api.github.com/gists/625fd039f07f678018cb6e0ccc81610c
#owner: https://api.github.com/users/rohitrajiit

import cv2
import os
import sqlite3
from ultralytics import YOLO
from PIL import Image
from deepface import DeepFace
from deepface.modules import verification
import numpy as np
import os
from PIL import Image

# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

#Load video
video_path = r'/Users/rohitraj/Downloads/PXL_20240224_164922392.TS.mp4'

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)


# Create/Connect to the Database
conn = sqlite3.connect('mydatabase.db') 
cursor = conn.cursor()

# Create a Table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        TIME INT
    )
''')

#save target embeddings
model_name = 'Facenet512'

testfolder = r'/Users/rohitraj/Downloads/test'
emlist = []
for file in os.listdir(testfolder):
    if '.jpg' in file or '.jpeg' in file:
        emlist.append([file.split('.')[0],DeepFace.represent(os.path.join(testfolder,file), model_name=model_name)[0]['embedding']])


def calldeepface(frame):
    try:
        inputembedding = DeepFace.represent(frame, model_name=model_name)[0]['embedding']
        for filename,targetembedding in emlist:
            distance = verification.find_cosine_distance(inputembedding, targetembedding)
            distance = np.float64(distance)
            threshold = verification.find_threshold(model_name, 'cosine')
            print(distance, threshold)
            if distance <= threshold:
                return filename
        return None
    except:
        return None


frame_count = 0
second_count = 0
oldperson = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % int(fps) == 0:
        results = model([frame],classes =[0], conf = 0.8)  # return a list of Results objects only for bus class with min confidence of 0.8
        if len (results)>0:
            # Process results list
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs
                if len(boxes)==0:
                    continue
                # img = Image.fromarray(frame)
                # crop = img.crop(boxes.xyxy[0].numpy())
                # person = calldeepface(np.array(crop))
                boxes = boxes.xyxy[0].numpy()
                crop = frame[int(boxes[1]):int(boxes[3]),int(boxes[0]):int(boxes[2])]
                person = calldeepface(crop)
                if person is not None and person !=oldperson:
                    cursor.execute('INSERT INTO attendance (name, TIME) VALUES (?, ?)', (person, second_count))
                    conn.commit()
                    oldperson = person
                    print(person, second_count)
        second_count += 1
        if second_count >100:
            break
        

    frame_count += 1

cap.release()