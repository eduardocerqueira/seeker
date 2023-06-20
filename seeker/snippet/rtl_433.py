#date: 2023-06-20T16:34:23Z
#url: https://api.github.com/gists/0ae794279af76c5f236c06ec0e68108c
#owner: https://api.github.com/users/jimduchek

from core.log import logging, LOG_PREFIX

#import core.utils

from java.lang import Thread, InterruptedException
import subprocess, select
import json
import signal, os, sys, time
import traceback

logger = logging.getLogger(LOG_PREFIX + ".rtl_433")


############## CONFIG HERE, EDIT THESE FOR SURE #######

RTL_433_COMMAND = "rtl_433 -R 40 -R 41 -R 55 -F json -C customary"


matches = { "Bedroom_Sensor": { "matches" : {"model":"Acurite-Tower",
                                          "id":14377},
                             "actions" : {"temperature_F":{"Bedroom_Temperature": None},
                                          "humidity"	 :{"Bedroom_Humidity":None},
                                          "battery_ok"	 :{"Bedroom_Temperature_Battery_Alarm":{"1":OFF,"0":ON}}
                                         }
                           },
             "Desk_Sensor": { "matches" : {"model":"Acurite-Tower",
                                          "id":7219},
                             "actions" : {"temperature_F":{"Desk_Temperature": None},
                                          "humidity"	 :{"Desk_Humidity":None},
                                          "battery_ok"	 :{"Desk_Temperature_Battery_Alarm":{"1":OFF,"0":ON}}
                                         }
                           },
             "Hallway_Sensor": { "matches" : {"model":"Acurite-Tower",
                                          "id":2506},
                             "actions" : {"temperature_F":{"Hallway_Temperature": None},
                                          "humidity"	 :{"Hallway_Humidity":None},
                                          "battery_ok"	 :{"Hallway_Temperature_Battery_Alarm":{"1":OFF,"0":ON}}
                                         }
                           },
             "Outside_Sensor": { "matches" : {"model":"Acurite-Tower",
                                          "id":14981},
                             "actions" : {"temperature_F":{"Outside_Temperature": None},
                                          "humidity"	 :{"Outside_Humidity":None},
                                          "battery_ok"	 :{"Outside_Temperature_Battery_Alarm":{"1":OFF,"0":ON}}
                                         }
                           },
             "Compartment_Sensor": { "matches" : {"model":"Acurite-606TX",
                                          "id":107},
                             "actions" : {"temperature_F":{"Compartment_Temperature": None},
                                          "battery_ok"	 :{"Compartment_Temperature_Battery_Alarm":{"1":OFF,"0":ON}}
                                         }
                           },
             "Freezer_Sensor": { "matches" : {"model":"Acurite-986",
                                          "id":19710},
                             "actions" : {"temperature_F":{"Freezer_Temperature": None},
                                          "battery_ok"	 :{"Freezer_Temperature_Battery_Alarm":{"1":OFF,"0":ON}}
                                         }
                           },
             "Refrigerator_Sensor": { "matches" : {"model":"Acurite-986",
                                          "id":11690},
                             "actions" : {"temperature_F":{"Refrigerator_Temperature": None},
                                          "battery_ok"	 :{"Refrigerator_Temperature_Battery_Alarm":{"1":OFF,"0":ON}}
                                         }
                           },
             "Crisper_Sensor": { "matches" : {"model":"Acurite-986",
                                          "id":7710},
                             "actions" : {"temperature_F":{"Crisper_Temperature": None},
                                          "battery_ok"	 :{"Crisper_Temperature_Battery_Alarm":{"1":OFF,"0":ON}}
                                         }
                           },
             "Chest_Freezer_Sensor": { "matches" : {"model":"Acurite-986",
                                          "id":24768},
                             "actions" : {"temperature_F":{"Chest_Freezer_Temperature": None},
                                          "battery_ok"	 :{"Chest_Freezer_Temperature_Battery_Alarm":{"1":OFF,"0":ON}}
                                         }
                           },
                           
          }

############## MORE CONFIG, PROBABLY DON'T NEED TO EDIT THESE #####

POLL_TIMEOUT=1000

############## END CONFIG


class RTL_433(Thread):
    def openRTL433(self):
        if self.rtl433_proc:
            self.rtl433_proc.kill()
            self.rtl433_proc = None
        try:
            self.rtl433_proc = subprocess.Popen(RTL_433_COMMAND.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except OSError:
            self.rtl433_proc = None
            self.logger.error("Unable to run '"+RTL_433_COMMAND+"'")
            self.interrupt()
            
    
    def runLoop(self):
        global matches
        self.rtl433_proc = None
        
        while not self.currentThread().isInterrupted():
            self.openRTL433()
            last_line = ''
            
            while not self.currentThread().isInterrupted():
                retcode = self.rtl433_proc.poll()
                if retcode is not None:
                    self.logger.info("RTL_433 Process Exited: %d" % retcode)
#                    if retcode == 1:
#                      subprocess.call("killall -9 rtl_433".split())
                    self.logger.info("Reset")
                    break
                    
                line = self.rtl433_proc.stdout.readline()
                if line == last_line:
                    continue
                last_line = line

            
                try:
                    jsonData = json.loads(line)
                except ValueError:
                    self.logger.info(line.rstrip())
                    continue #ignore non-JSON
                
                sentEvent = False
                
                self.logger.debug(line.rstrip())

                for name, objData in matches.iteritems():
                    matched = True
                    for key, value in objData["matches"].iteritems():
                        if key in jsonData and str(value) == str(jsonData[key]):
                            continue
                        else:
                            matched = False
                            break
                    
                    if not matched:
                        continue 
                    
                    for key, value in objData["actions"].iteritems():
                        if not key in jsonData:
                            self.logger.info("Can't find key: "+key)
                            continue
                        for item, action in value.iteritems():
                            if action is None: # Send it straight on
                                sentEvent = True
                                events.postUpdate(str(item), str(jsonData[key]))
                            else:
                                for v, mapper in action.iteritems():
                                    if jsonData[key] == v:
                                        if items[str(item)] != mapper:
                                            sentEvent = True
                                            events.sendCommand(str(item), str(mapper))

                if not sentEvent:
                    self.logger.debug("No event sent: "+line.rstrip())
                        
                        
                
            if not self.currentThread().isInterrupted():
                self.logger.info("Restarting RTL_433 in 5 seconds")
                time.sleep(5)
                continue
                
        self.logger.error("Runloop out!")
                
        if self.rtl433_proc:
            self.rtl433_proc.kill()
            self.rtl433_proc = None

    def run(self):        
        self.logger = logging.getLogger(LOG_PREFIX + ".rtl_433")
        
        try:
            self.runLoop()
        except ClosedByInterruptException:
            pass
        except Exception as e:
            self.logger.error("Caught Exception: "+str(e))
            self.logger.error(traceback.format_exc())         

        self.logger.info("Did we get here?")            
        if self.rtl433_proc:
            self.logger.info("Did we get here too?")
            self.rtl433_proc.kill()





rtl433 = RTL_433()


def scriptLoaded(id):
    global rtl433
    rtl433.start() 
    
    
def scriptUnloaded():
    global rtl433
    logger.info("Unloading, and I wait")
    rtl433.interrupt()
    rtl433.join()
    logger.info("Unloaded")

