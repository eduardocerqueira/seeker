#date: 2022-02-10T16:45:19Z
#url: https://api.github.com/gists/e9ce11a4eb49a6ae94efac74b2dae36e
#owner: https://api.github.com/users/elbruno

#    Copyright (c) 2022
#    Author      : Bruno Capuano
#    Create Time : 2022 Feb
#    Change Log  :
#    - Open a camera feed from a local webcam and analyze each frame to detect faces using haar cascades
#    - When a face is detected, the app will blur the face zone
#    - Press [Q] to quit the app
#
#    The MIT License (MIT)
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in
#    all copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#    THE SOFTWARE.

import cv2
import time

video_capture = cv2.VideoCapture(0)
time.sleep(2)

# enable face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# open
while True:
    try:
        _, frameOrig = video_capture.read()
        frame = cv2.resize(frameOrig, (640, 480))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (top, right, bottom, left) in faces:
            cv2.rectangle(frame,(top,right),(top+bottom,right+left),(0,0,255),2)
            
            face = frame[right:right+left, top:top+bottom]
            face = cv2.GaussianBlur(face,(23, 23), 30)

            # merge this blurry rectangle to our final image
            frame[right:right+face.shape[0], top:top+face.shape[1]] = face

        cv2.imshow('@elbruno - Face Blur', frame)

    except Exception as e:
        print(f'exc: {e}')
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()