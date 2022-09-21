#date: 2022-09-21T17:24:28Z
#url: https://api.github.com/gists/06b28b0f8de7ecb14790dd31c5525b2d
#owner: https://api.github.com/users/liquid182

import os
import re

def isCamOn():
    stream = os.popen("log show --last 90m --predicate 'subsystem contains \"com.apple.UVCExtension\" and composedMessage contains \"Post PowerLog\"'")
    output = stream.readlines()
    regex=".*\"VDCAssistant_Power_State\"\s=\s(On|Off)";
    webcamIsOn=False
    for item in output:
        match = re.search(regex,item)
        if match == None:
            continue
        elif match.group(1) == "On":
            webcamIsOn=True
        else:
            webcamIsOn=False
    return webcamIsOn

def isMicOn():
    stream = os.popen("/usr/sbin/ioreg -l | grep -o  \"\\\"IOAudioEngineState\\\" = 1\" | wc -l")
    output = stream.read()
    micIsOn=False
    if( int(output) > 0 ):
        micIsOn=True
    return micIsOn
