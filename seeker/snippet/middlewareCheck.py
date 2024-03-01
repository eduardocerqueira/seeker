#date: 2024-03-01T16:49:17Z
#url: https://api.github.com/gists/13081e2933938d13e2047f15b2c9f007
#owner: https://api.github.com/users/LearningIsJourney

# WebLogic 12c health monitoring tool
# Developer: Gian Luca Ricci
# Just another Pythoneer
# Code version: 4.0.1
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
# Beautiful is better than ugly.
# Explicit is better than implicit.
# Simple is better than complex.
# Complex is better than complicated.
# Flat is better than nested.
# Sparse is better than dense.
# Readability counts.
# If the implementation is hard to explain, it's a bad idea.
# There should be one-- and preferably only one --obvious way to do it.

import os
import sys
from os import listdir
from java.io import FileInputStream
import weblogic.security.internal.SerializedSystemIni
import weblogic.security.internal.encryption.ClearOrEncryptedService
from tempfile import mktemp

# You can modify only this CONSTANT
DIR_FILE_PROPERTIES='/home/oracle/MonitoraggiOracleToTivoli/ConfigFilesDir/'

# If the script fails, please comment the following line, so you could see the DumpStack Trace.
redirect('/dev/null','false')

print '*** Script invocation : Starting ***'

listConfigFiles = list()

def getPropertyFiles(propFilesPath):
    try:
        for f in listdir(propFilesPath):
            if f.endswith('.properties'):
                listConfigFiles.append(f)
    except NotADirectoryError:
        print 'Error: not a directory'
        exit(exitcode=1)
    except PermissionError:
        print 'Permission deny'
        exit(exitcode=2)
    except:
        print 'Unexpected error'
        exit(exitcode=3)

getPropertyFiles(DIR_FILE_PROPERTIES)

for propertiesFile in listConfigFiles:
    try:
        propertiesFile = DIR_FILE_PROPERTIES + propertiesFile
        print 'Reading Config File: ' + propertiesFile
        readDomainProperties = FileInputStream(propertiesFile)
        configProps = Properties()
        configProps.load(readDomainProperties)

        admin_server = configProps.get("wls.admin_server")
        userConfig = configProps.get("wls.userConfiFile")
        userKey = configProps.get("wls.userKeyFile")
        logFile = configProps.get("logger.file")
    except IOError:
        print("Can 't open: " + propertiesFile)
        exit(exitcode=4)
    except:
        dumpStack()
        print("Unexpected error")
        exit(exitcode=5)

    domainGlobalInfo = {}
    completeServersList = ()
    shutDownList = []
    try:
        user = os.path.abspath(userConfig)
        key = os.path.abspath(userKey)
        print 'Connecting to: ' + admin_server + ' with user: ' + user + ' and key: ' + key
        connect(userConfigFile=user,userKeyFile=key,url=admin_server)
        wlsVersion = version
        completeServersList = ls('Servers',returnMap='true')
        domainRuntime()
        domainName = cmo.getName()
        for singleServer in completeServersList:
            serverLife = cmo.lookupServerLifeCycleRuntime(singleServer)
            serverStatus = serverLife.getState()
            if str(serverStatus) == "SHUTDOWN" and str(singleServer).lower() != "adminserver":
                shutDownList.append(str(singleServer) + ";" + str(domainName) + ";None;" + str(serverStatus) + ";0;0;0;0;0;0")
        completeServersList = ()
        cd('ServerRuntimes')

        servers=domainRuntimeService.getServerRuntimes()
        for server in servers :
            hogging = 0
            stuck = 0
            serverName = server.getName()
            serverState = server.getState()
            serverHealt = server.getHealthState()
            serverHost = server.getListenAddress()
            freeHeap = int(server.getJVMRuntime().getHeapFreeCurrent())/(1024*1024)
            freeHeapPerc = int(server.getJVMRuntime().getHeapFreePercent())
            currentHeap = int(server.getJVMRuntime().getHeapSizeCurrent())/(1024*1024)
            heapMax = int(server.getJVMRuntime().getHeapSizeMax())/(1024*1024)
            if wlsVersion[:2] == "12":
                cd(serverName + '/ThreadPoolRuntime/ThreadPoolRuntime')
                hogging = get('HoggingThreadCount')
                stuck = get('StuckThreadCount')
                cd('/ServerRuntimes/')
            # This else block is for WebLogic 11g, because ThreadPoolRuntime Mbean 
            # listing is missing StuckThreadCount value in ServerRuntime from WLST.
            # So you have to apply this patch: 14466837(and remove the else statement)
            # or use this script with thi work around.
            else:
                tempDump = mktemp()
                threadDump(writeToFile="true", serverName=serverName, fileName=tempDump)
                f = open(tempDump, "r")
                for i in f.readlines():
                    if i.find("STUCK") > 0:
                        stuck += 1
                f.close()
                os.remove(tempDump)
                cd(serverName + '/ThreadPoolRuntime/ThreadPoolRuntime')
                hogging = get('HoggingThreadCount')
                cd('/ServerRuntimes/')

            if str(server.getName()).lower() != 'adminserver':
                domainGlobalInfo[str(server.getName())] = str(domainName) + ";" + str(serverHost) + ";" + str(serverState) + ";" + str(hogging) + ";" + str(stuck) + ";" + str(heapMax) + ";" + str(currentHeap) + ";" + str(freeHeap) + ";" + str(freeHeapPerc) + "%"
    except NameError:
        print('WebLogic username or password are wrong')
        #exit(exitcode=6)
    except:
        dumpStack()
        print('Unexpected error')
        #exit(exitcode=7)

    try:
        print 'Logging...'
        myLogger = open(logFile,"a")
        for k, v in domainGlobalInfo.iteritems():
            myLogger.write(str(k) + ";" + str(v) + "\n")
            print k, v
        if len(shutDownList) > 0:
            for shutDownServer in shutDownList:
                print shutDownServer
                myLogger.write(shutDownServer + "\n")
        myLogger.close()
    except:
        myLogger.close()
        print("Can't write log in: " + logFile)
        exit(exitcode=8)

    disconnect()

exit(exitcode=0)