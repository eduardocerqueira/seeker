#date: 2021-12-21T16:54:56Z
#url: https://api.github.com/gists/7940cb69b040347916da480f94b86541
#owner: https://api.github.com/users/kristipatithoyajakshakashyap

import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import time

class Button:
    def __init__(self, pos, width, height, value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value

    def draw(self,img):
        cv.rectangle(img, self.pos, (self.pos[0]+self.width, self.pos[1]+self.height), (225, 225, 225), cv.FILLED)
        cv.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), (50, 50, 50), 3)
        cv.putText(img, self.value, (self.pos[0]+ 37, self.pos[1] + 60), cv.FONT_HERSHEY_PLAIN, 2, (50, 50, 50), 2)
    def checkClick(self, x, y):
        if  self.pos[0]<x<self.pos[0] + self.width and self.pos[1]<y<self.pos[1] + self.height:
            cv.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), (255, 255, 255),
                         cv.FILLED)
            cv.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), (50, 50, 50), 3)
            cv.putText(img, self.value, (self.pos[0] + 20, self.pos[1] + 70), cv.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 5)
            return True
        else:
            return False

#webcame
cap = cv.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720) # height
detector = HandDetector(detectionCon=0.8, maxHands=1)

#creating buttons
buttonListValues = [['7','8','9', '*'],
                    ['4','5','6', '-'],
                    ['1','2','3', '+'],
                    ['0','/','.', '=']]

buttonList = []
for x in range(4):
    for y in range(4):
        xpos = x * 100 + 800
        ypos = y * 100 + 170
        buttonList.append(Button((xpos,ypos), 100, 100, buttonListValues[y][x]))

# Variable
myEquation = ""
delayCounter = 0

#loop
while True: 
    success, img = cap.read()
    img = cv.flip(img, 1)

    # Detection of hand
    hands, img= detector.findHands(img, flipType=False)
    
    #Draw all btns
    # Text
    cv.rectangle(img, (800, 50), (800 + 400, 70 + 100), (225, 225, 225), cv.FILLED)
    cv.rectangle(img,  (800, 50), (800 + 400, 70 + 100), (50, 50, 50), 3)
    # btns
    for btn in buttonList:
        btn.draw(img)

    # Check for Hand
    if hands:
        lmList = hands[0]['lmList']
        length, _, img =  detector.findDistance(lmList[8],lmList[12], img)
        x, y= lmList[8]
        if length < 60:
            for i, btn in enumerate(buttonList):
               if  btn.checkClick(x,y) and delayCounter == 0:
                   myValue= buttonListValues[int(i%4)][int(i/4)]
                   if myValue == '=':
                       myEquation = str(eval(myEquation))
                   else:
                       myEquation += myValue
                   delayCounter = 1

    # Avoid Duplicates
    if delayCounter != 0 :
        delayCounter += 1
        if delayCounter > 10:
            delayCounter = 0

    #Display the Equation
    cv.putText(img, myEquation, (810, 120), cv.FONT_HERSHEY_PLAIN, 3, (50, 50, 50), 3 )

    # Display img
    cv.imshow("Image", img)
    key = cv.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    
    if key == ord('c'):
        myEquation=''