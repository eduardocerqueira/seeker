#date: 2021-11-10T17:07:28Z
#url: https://api.github.com/gists/e658cceffef3ee9cc049799d72a7cc8a
#owner: https://api.github.com/users/Leo-PL

# Enable "charge + tether" in Settings -> Setup -> Mobile Router Setup -> Tethering
# Use a usb cable to connect (tethering) to the router on port 5510

# Useful links:
# Verify the IMEI number
# https://en.wikipedia.org/wiki/Luhn_algorithm
# Challenge/Response Generator for Sierra Wireless Cards V1.0 
# https://github.com/bkerler/SierraWirelessGen

ATI
  Manufacturer: Netgear, Incorporated
  Model: MR1100
  Revision: NTG9X50C_12. ...
  IMEI: 987654321098765
  IMEI SV: 99
  FSN: BBBBBBBBBBBBB
  +GCAP: +CGSM 
  OK
  
AT!OPENLOCK?
  AT!OPENLOCK?
  AAAAAAAAAAAAAAAA
  OK

# Compute the response with SierraWirelessGen
# ./sierrakeygen.py -l 'AAAAAAAAAAAAAAAA' -d 'MDM9x50_V1'
# AT!OPENLOCK="0480E46C7E30F561"

AT!OPENLOCK="0480E46C7E30F561"
  AT!OPENLOCK="0480E46C7E30F561"
  OK

AT!NVIMEIUNLOCK
  AT!NVIMEIUNLOCK
  OK
 
AT!NVENCRYPTIMEI=12,34,56,78,90,12,34,52
  AT!NVENCRYPTIMEI=12,34,56,78,90,12,34,52
  OK

AT!RESET
  AT!RESET
  OK
