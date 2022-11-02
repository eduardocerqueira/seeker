#date: 2022-11-02T17:23:23Z
#url: https://api.github.com/gists/c4e1b88168bdc6170fe95c5068859663
#owner: https://api.github.com/users/SULAIMAN-5-AHMED

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
while True:
    successs, img = cap.read()
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRgb)

    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id,cx,cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 30, (255, 255, 255), cv2.FILLED)

            mpdraw.draw_landmarks(img, handlms, mphands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 256, 0))
    cv2.imshow('Image', img)
    cv2.waitKey(1)