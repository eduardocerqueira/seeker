#date: 2023-06-21T17:08:38Z
#url: https://api.github.com/gists/5ccc00a773657cf22f8a9a58781fdc77
#owner: https://api.github.com/users/caternuson

import time
import pyvisa

RIGOL = 'USB0::6833::1230::DS1ZA191806168::0::INSTR'
FILE = "rigol_capture.png"

rm = pyvisa.ResourceManager()
rigol = rm.open_resource(RIGOL, write_termination='\n', read_termination='\n')
time.sleep(3) # pause to let "USB Device Connected" dialog to clear
raw_buf = rigol.query_binary_values(':DISP:DATA? ON,0,PNG', datatype='B')

with open(FILE, "wb") as fp:
    fp.write(bytearray(raw_buf))

print("Screen saved to", FILE)

rigol.close()