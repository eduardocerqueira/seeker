#date: 2022-09-23T17:10:20Z
#url: https://api.github.com/gists/12e0ea254deca93d8ddae9363854ef3c
#owner: https://api.github.com/users/OhadRubin

analog_keys = {0:0, 1:0, 2:0, 3:0, 5:0}
import numpy as np

# START OF GAME LOOP
while running:
    state = np.array(mycobot.get_angles())
    if len(state)!=6:
        state = np.zeros(6)
    ################################# CHECK PLAYER INPUT #################################
    for event in pygame.event.get():
        delta = np.zeros(6)
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            ############### UPDATE SPRITE IF SPACE IS PRESSED #################################
            pass
        changed = False
        # HANDLES BUTTON PRESSES
        if event.type == pygame.JOYBUTTONDOWN:
            if event.button == button_keys['left_arrow']:
                delta[0] -= 1
                changed = True
            if event.button == button_keys['right_arrow']:
                delta[0] += 1
                changed = True
            if event.button == button_keys['down_arrow']:
                delta[1] -= 1
                changed = True
            if event.button == button_keys['up_arrow']:
                delta[1] += 1
                changed = True
                
        if event.type == pygame.JOYAXISMOTION:
            analog_keys[event.axis] = event.value
            
            if abs(analog_keys[0]) > .4:
                if  analog_keys[0] < -.7:
                    delta[2] += 1
                    changed = True
                if  analog_keys[0] > .7:
                    delta[2] -= 1
                    changed = True
            if abs(analog_keys[1]) > .4:
                if  analog_keys[1] < -.7:
                    delta[3] += 1
                    changed = True
                if  analog_keys[1] > .7:
                    delta[3] -= 1
                    changed = True
            if abs(analog_keys[2]) > .4:
                if  analog_keys[2] < -.7:
                    delta[4] += 1
                    changed = True
                if  analog_keys[2] > .7:
                    delta[4] -= 1
                    changed = True
                changed = True
            if abs(analog_keys[5]) > .4:
                if  analog_keys[5] < -.7:
                    delta[5] += 1
                    changed = True
                if  analog_keys[5] > .7:
                    delta[5] -= 1
                    changed = True
                


    state += delta
    state = np.clip(state,-180,180)
    movement_constant = 15
    if changed:
        mycobot.send_angles(state, 80)