#date: 2021-12-27T16:45:06Z
#url: https://api.github.com/gists/30908465ce93c84100d467eb21764f7b
#owner: https://api.github.com/users/Park-Developer

import cv2

cap=cv2.VideoCapture(0) # 카메라 0번 장치 연결

width=cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 프레임 폭
height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 프레임 높이
print("Original Size",width,height)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
width=cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 프레임 폭
height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 프레임 높이
print("Resized Size",width,height)

if cap.isOpened():
    while True:
        ret, img=cap.read()

        if ret:
            cv2.imshow("Test",img)
            if cv2.waitKey(1)!=-1: #1ms동안 키 입력 대기 =>아무키나 누르면 중지
                break
        else:
            break
else:
    print("Can't Open Video")


cap.release() # 캡쳐 자원 반납납
cv2.detroyAllWindows()