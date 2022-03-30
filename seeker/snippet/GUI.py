#date: 2022-03-30T16:51:52Z
#url: https://api.github.com/gists/b38cb3e857620ef58e1027db210f1777
#owner: https://api.github.com/users/skarantha22

from guizero import App, PushButton, Text
from gpiozero import LED
import sys
import time 
import RPi.GPIO as GPIO 
import paho.mqtt.client as mqtt

led = LED(21)
led1 = LED(22)
led3 = LED(23)

def exitApp():
    sys.exit()
    
def toggleLED():
    led.toggle()
    if led.is_lit :
        ledButton.text ="MOTOR1 OFF"
        welcome_message.value ="MOTOR1_OFF";
        client = mqtt.Client()
        client.connect('localhost', 1883, 60) 
        client.loop_start()
        print('Script is running, press Ctrl-C to quit...') 
        if True:
            client.publish('motor1_onoff', '0')
    else :
        ledButton.text ="MOTOR1 ON"
        welcome_message.value ="MOTOR1_OFF";
        client = mqtt.Client()
        client.connect('localhost', 1883, 60) 
        client.loop_start()
        print('Script is running, press Ctrl-C to quit...') 
        if True:
            client.publish('motor1_onoff', '1')
def toggleLED1():
    led1.toggle()
    if led1.is_lit :
        ledButton1.text ="MOTOR2 OFF"
        welcome_message.value ="MOTOR2_oFF";
        client = mqtt.Client()
        client.connect('localhost', 1883, 60) 
        client.loop_start()
        print('Script is running, press Ctrl-C to quit...') 
        if True:
            client.publish('motor2_onoff', '0')
    else :
        ledButton1.text ="MOTOR2 ON"
        client = mqtt.Client()
        client.connect('localhost', 1883, 60) 
        client.loop_start()
        print('Script is running, press Ctrl-C to quit...') 
        if True:
            client.publish('motor2_onoff', '1')
            
def toggleLED2():
    led1.toggle()
    if led1.is_lit :
        ledButton2.text ="MOTOR1 DIR_R"
        client = mqtt.Client()
        client.connect('localhost', 1883, 60) 
        client.loop_start()
        print('Script is running, press Ctrl-C to quit...') 
        if True:
            client.publish('motor1_dir', '0')
    else :
        ledButton2.text ="MOTOR1 DIR_L"
        client = mqtt.Client()
        client.connect('localhost', 1883, 60) 
        client.loop_start()
        print('Script is running, press Ctrl-C to quit...') 
        if True:
            client.publish('motor1_dir', '1')
            
def toggleLED3():
    led1.toggle()
    if led1.is_lit :
        client = mqtt.Client()
        client.connect('localhost', 1883, 60) 
        client.loop_start()
        print('Script is running, press Ctrl-C to quit...') 
        if True:
            client.publish('motor_speed', '0')
    else :
        motor2DirectionControlButton.text ="MOTOR2 DIR_L"
        welcome_message.value ="MOTOR2 DIR_L";
        client = mqtt.Client()
        client.connect('localhost', 1883, 60) 
        client.loop_start()
        print('Script is running, press Ctrl-C to quit...') 
        if True:
            client.publish('motor2_dir', '1')

def toggleLED4():
    led3.toggle()
    if led3.is_lit :
        motorSpeedControlButton.text="MOTOR_SPEED_H"
        client = mqtt.Client()
        client.connect('localhost', 1883, 60) 
        client.loop_start()
        print('Script is running, press Ctrl-C to quit...') 
        if True:
            client.publish('motorspeed', '0')
    else :
        motorSpeedControlButton.text ="MOTOR_SPEED_L"
        client = mqtt.Client()
        client.connect('localhost', 1883, 60) 
        client.loop_start()
        print('Script is running, press Ctrl-C to quit...') 
        if True:
            client.publish('motorspeed', '1')
    
    
app = App('First Gui', height = 600, width = 800)

welcome_message = Text(app, text="Welcome to my app")

ledButton = PushButton(app, toggleLED, text="MOTOR1 ON", align="top",width = 10, height = 2)
ledButton.text_size = 10

ledButton1 = PushButton(app, toggleLED1, text="MOTOR2 ON", align="top",width = 10, height = 2)
ledButton1.text_size = 10

ledButton2 = PushButton(app, toggleLED2, text="MOTOR1 DIR_R", align="top",width = 10, height = 2)
ledButton2.text_size = 10

motor2DirectionControlButton = PushButton(app, toggleLED3, text="MOTOR2 DIR_R", align="top",width = 10, height = 2)
motor2DirectionControlButton.text_size = 10

motorSpeedControlButton = PushButton(app, toggleLED4, text="MOTOR_SPEED_H", align="top",width = 15, height = 2)
motorSpeedControlButton.text_size = 10


exitButton = PushButton(app, exitApp, text="Exit", align="bottom" , width = 5, height = 2)
exitButton.text_size = 20

app.display()

