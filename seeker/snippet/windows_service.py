#date: 2022-05-03T17:05:00Z
#url: https://api.github.com/gists/5d9f221dc51961f090305856c86c5284
#owner: https://api.github.com/users/kkldream

import win32serviceutil
import win32service
import win32event
import os
import logging
import inspect
import time

def service_main():
    print_log('time is', int(time.time()))
    time.sleep(1)

class PythonService(win32serviceutil.ServiceFramework):
    _svc_name_ = "kk_service_test"
    _svc_display_name_ = "kk_service_test"
    _svc_description_ = "test"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.logger = self._getLogger()
        self.run = True

    def _getLogger(self):
        logger = logging.getLogger(f'[{self._svc_name_}]')
        this_file = inspect.getfile(inspect.currentframe())
        dirpath = os.path.abspath(os.path.dirname(this_file))
        handler = logging.FileHandler(os.path.join(dirpath, "service.log"))
        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def SvcDoRun(self):
        global print_log
        self.logger.info("service is run....")
        def print_generate(*objects, sep=' ', end='\n'):
            if len(objects) > 0:
                msg = objects[0]
                if len(objects) > 1:
                    for obj in objects[1:]:
                        msg += sep + str(obj)
                self.logger.info(msg)
            print(*objects, sep=sep, end=end)
        print_log = print_generate
        while self.run:
            service_main()

    def SvcStop(self):
        self.logger.info("service is stop....")
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.run = False

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(PythonService)