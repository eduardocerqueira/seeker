#date: 2024-02-29T17:03:09Z
#url: https://api.github.com/gists/9c9771bff16d76f8a763ba47bec65b81
#owner: https://api.github.com/users/layeddie

#!/usr/bin/python3
import notecard
from notecard import hub
from periphery import I2C
import cv2
import keys
import time
from picamera2 import Picamera2
from smbus2 import SMBus
from bme280 import BME280

# init the notecard
productUID = keys.NOTEHUB_PRODUCT_UID
port = I2C("/dev/i2c-1")
nCard = notecard.OpenI2C(port, 0, 0)

# connect notecard to notehub
rsp = hub.set(nCard, product=productUID, mode="continuous")
print(rsp)

# create note template
req = {"req": "note.template"}
req["file"] = "face.qo"
req["body"] = {"face_count": 11, "voltage": 12.1, "temperature": 12.1, "pressure": 12.1, "humidity": 12.1}
rsp = nCard.Transaction(req)

# init the BME280
bus = SMBus(1)
bme280 = BME280(i2c_dev=bus, i2c_addr=0x77)

# Grab images as numpy arrays and leave everything else to OpenCV.
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cv2.startWindowThread()

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# keep track of face counts between notes
face_count = 0

# keep track of seconds for adding faces/syncing
start_secs_face = int(round(time.time()))
start_secs_note = int(round(time.time()))

def send_note(c):
   
    # query the notecard for power supply voltage
    req = {"req": "card.voltage", "mode": "?"}
    rsp = nCard.Transaction(req)
    voltage = rsp["value"]
   
    # get the temp/pressure/humidity from bme280
    temperature = bme280.get_temperature()
    pressure = bme280.get_pressure()
    humidity = bme280.get_humidity()

    req = {"req": "note.add"}
    req["file"] = "face.qo"
    req["body"] = {"face_count": c, "voltage": voltage, "temperature": temperature, "pressure": pressure, "humidity": humidity}
    req["sync"] = True
    rsp = nCard.Transaction(req)

    print(rsp)

while True:
    # track the current time
    current_seconds = int(round(time.time()))
   
    im = picam2.capture_array()

    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(grey, 1.1, 5)

    # Add text around each face
    font = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2
   
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0))
        face_plural = 's'
        if face_count == 1:
            face_plural = ''
        cv2.putText(im, 'Face found!', (x, y-10), font,
                   fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Camera", im)
   
    if len(faces) > 0:
        # check to make sure it's been at least three seconds since the last time we checked for faces
        if current_seconds - start_secs_face >= 3:
            face_count += len(faces)
            print("We found some faces: " + str(len(faces)) + " to be exact! (Pending sync: " + str(face_count) + " faces)")
            start_secs_face = int(round(time.time()))
   
    # create an outbound note every 5 minutes with accumulated face counts
    if current_seconds - start_secs_note >= 60:
        send_note(face_count)
        print("####################")
        print("Sending a new note with " + str(face_count) + " faces.")
        print("####################")
        face_count = 0
        start_secs_note = int(round(time.time()))
   
    cv2.waitKey(1)