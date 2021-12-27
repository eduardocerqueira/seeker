#date: 2021-12-27T17:19:13Z
#url: https://api.github.com/gists/50d20c0d9e47f002d5384666327c9932
#owner: https://api.github.com/users/ajay-rawat-9

import face_recognition
import cv2
import numpy as np
import glob
import pickle
f=open("ref_name.pkl","rb")
ref_dictt=pickle.load(f)        
f.close()

f=open("ref_embed.pkl","rb")
embed_dictt=pickle.load(f)      
f.close()

known_face_encodings = []  
known_face_names = []

for ref_id , embed_list in embed_dictt.items():
    for my_embed in embed_list:
        known_face_encodings +=[my_embed]
        known_face_names += [ref_id]

import time,json,conf
from boltiot import Bolt
mydev=Bolt(conf.api_dev,conf.dev_id)
def read():
        print("Reading LDR value")
        input=mydev.analogRead('A0')
        data=json.loads(input)
        print("LDR value is: "+str(data['value']))
        return data
def buzzon():
        output=mydev.digitalWrite('0','HIGH')
        print(output)
def buzzoff():
        output=mydev.digitalWrite('0','LOW')
        print(output)
def motoron():
        output=mydev.digitalWrite('1','HIGH')
        print(output)
def motoroff():
        output=mydev.digitalWrite('1','LOW')
        print(output)
data1=read()
val1=int(data1['value'])
Found=False
Unknown=False
while True:
        time.sleep(5)
        data2=read()
        val2=int(data2['value'])
        val1=val1+40
        if val2<val1:
                video_capture = cv2.VideoCapture(0)
                face_locations = []
                face_encodings = []
                face_names = []
                process_this_frame = True
                while True:
                        ret, frame = video_capture.read()
                        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                        rgb_small_frame = small_frame[:, :, ::-1]
                        if process_this_frame:
                                face_locations = face_recognition.face_locations(rgb_small_frame)
                                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                                face_names = []
                                for face_encoding in face_encodings:
                                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                                        name = "Unknown"
                                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                        best_match_index = np.argmin(face_distances)
                                        if matches[best_match_index]:
                                                print("Face is found")
                                                name = known_face_names[best_match_index]
                                                print("Welcome",ref_dictt[name])
                                                motoron()
                                                time.sleep(2)
                                                motoroff()
                                                Unknown=False
                                                Found=True
                                        elif name=="Unknown":
                                                print("face didn't matched!!!!!")
                                                motoroff()
                                                buzzon()
                                                time.sleep(2)
                                                buzzoff()
                                                Unknown=True
                                                Found=False
                                        face_names.append(name)
                                        video_capture.release()
                                        cv2.destroyAllWindows()
                                        if Unknown or Found:
                                                break
                                if Unknown or Found:
                                        break        
                if Unknown or Found:
                        break
