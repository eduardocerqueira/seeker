#date: 2023-08-25T17:01:49Z
#url: https://api.github.com/gists/094754bc3798b4a0fdae00bd8f2e6a71
#owner: https://api.github.com/users/nullpepe

#!/bin/bash

#Takes filename from path between /<filename>:
#Then it takes _p and converts to :
#Lastly it removes any junk after the port number
extract="sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/'"


#Create Output Directory
echo "Creating Directories..."
mkdir -p "parsed_testssl"
sleep 1
echo

#Moving Failed/Empty TestSSL Outputs
echo "Moving Failed/Empty Outputs..."
sleep 1
echo
mkdir -p "empty_files" && find . -maxdepth 1 -size -1000c -exec mv {} empty_files/ \;


echo "Parsing TestSSL Outputs..."
sleep 1
echo

###Testing Protocols###
#echo "Parsing Testing Protocols"
grep -d skip -E "SSLv2.*NOT" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/SSLv2_Offered.txt
grep -d skip -E "SSLv3.*NOT" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/SSLv3_Offered.txt
grep -d skip -E "TLS 1.*deprecated" "$PWD/"* | grep -v "TLS 1.1" | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/TLSv1.0_Offered.txt
grep -d skip -E "TLS 1.1.*deprecated" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/TLSv1.1_Offered.txt

###Testing Ciphers###
#echo "Parsing Testing Ciphers"
grep -d skip -E "NULL ciphers.*NOT" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/NULL_Ciphers_Offered.txt
grep -d skip -E "Anonymous NULL.*NOT" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Anonymous_NULL_Ciphers_Offered.txt
grep -d skip -E "Export ciphers.*NOT" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Export_Ciphers_Offered.txt
grep -d skip -E "LOW.*NOT" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/LOW_Ciphers_Offered.txt
grep -d skip -E "Triple DES.*offered" "$PWD/"* | grep -v "not" | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/3DES_IDEA_Ciphers_Offered.txt
grep -d skip -E "Obsoleted CBC.*offered" "$PWD/"* | grep -v "not" | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Obsolete_CBC_Ciphers_Offered.txt
grep -d skip -E "Strong encryption.*not offered" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Strong_Encryption_no_FS_Not_Offered.txt
grep -d skip -E "Forward Secrecy strong.*not offered" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Forward_Secrecy_Strong_Encryption_Not_Offered.txt

###Testing server's cipher preferences###
#echo "Parsing Testing server's cipher preferences"
#grep -d skip -E "cipher order.*NOT" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/No_Server_Cipher_Order.txt
#sleep 1
#echo

###Testing robust forward secrecy (FS)###
#echo "Parsing Testing robust forward secrecy (FS)"
#grep -d skip -E "supporting Forward Secrecy" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/No_Ciphers_Supporting_FS_Offered.txt
grep -d skip -E "DH group offered.*1024" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/DH_Group_Offered_1024_bits.txt
grep -d skip -E "DH group offered.*768" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/DH_Group_Offered_768_bits.txt

###Testing server defaults###
#echo "Parsing Testing server defaults"
grep -d skip -E "Signature Algorithm.*SHA1 with RSA" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Signature_Algorithm_SHA1_with_RSA.txt
grep -d skip -E "Signature Algorithm.*MD5" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Signature_Algorithm_MD5.txt
grep -d skip -E "Server key size.*1024" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Server_Key_Size_1024_bits.txt
grep -d skip -E "Server key size.*512" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Server_Key_Size_512_bits.txt
#grep -d skip -E "subjectAltName.*missing" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/subjectAltName_Missing.txt
grep -d skip -oE "certificate does not match supplied URI" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Certificate_does_not_match_supplied_URI.txt
grep -d skip -E "Chain of trust.*NOT.*self signed" "$PWD/"* | grep -v "CA" | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Chain_of_Trust_Self_Signed_Cert.txt
#grep -d skip -E "Chain of trust.*NOT.*CA" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Chain_of_Trust_Self_Signed_CA.txt
grep -m 1 -d skip -E "Chain of trust.*chain incomplete" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Chain_of_Trust_Incomplete.txt
grep -d skip -E "Certificate Validity.*expired" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Certificate_Expired.txt
#grep -d skip -E "Certificate Validity.*expires < 60" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Certificate_Expiring_60_Days_or_less.txt
#grep -d skip -E "CRL nor OCSP" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/CRL_nor_OCSP_URI_provided.txt
#grep -d skip -E "OCSP stapling.*not offered" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/OSCP_Stapling_Not_Offered.txt
#grep -d skip -E "DNS CAA RR.*not offered" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/DNS_CAA_RR_Not_Offered.txt

###Testing HTTP header response @ "/"###
#echo "Parsing Testing HTTP header response @ /"
grep -d skip -E "HTTP Status Code.*insecure" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Redirects_to_insecure_URL.txt
#grep -d skip -E "Strict Transport Security.*not offered" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/STS_Not_Offered.txt
#grep -d skip -E "Strict Transport Security.*too short" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/STS_Too_Short.txt
grep -d skip -E "Cookie.*NOT" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Secure_Cookie_Flags_Missing.txt


###Testing vulnerabilities###
#echo "Parsing Testing vulnerabilities"
#grep -d skip -E "Heartbleed.NOT" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Heartbleed_CVE-2014-0160.txt
#grep -d skip -E "CCS.*VULNERABLE" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/CCS_CVE-2014-0224.txt
#grep -d skip -E "Ticketbleed.*NOT" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Ticketbleed_CVE-2016-9244.txt
#grep -d skip -E "ROBOT.*VULNERABLE" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/ROBOT.txt
grep -d skip -E "Secure Renegotiation.*VULNERABLE" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Secure_Renegotiation-RFC_5746_Not_Supported.txt
grep -d skip -E "Client-Initiated.*VULNERABLE" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Secure_Client-Initiated_Renegotiation-DoS_Threat.txt
#grep -d skip -E "CRIME.*VULNERABLE" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/CRIME_CVE-2012-4929.txt
#grep -d skip -E "BREACH.*NOT" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/BREACH_CVE-2013-3587.txt
#grep -d skip -E "POODLE.*VULNERABLE" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/POODLE_CVE-2014-3566.txt
#grep -d skip -E "TLS_FALLBACK_SCSV.*prevention NOT supported" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/TLS_FALLBACK_SCSV_RFC-7507.txt
#grep -d skip -E "SWEET32.*VULNERABLE" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/SWEET32_CVE-2016-2183_CVE-2016-6329.txt
#grep -d skip -E "FREAK.*VULNERABLE" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/FREAK_CVE-2015-0204.txt
#grep -d skip -E "DROWN.*VULNERABLE" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/DROWN_CVE-2016-0800_CVE-2016-0703.txt
#grep -d skip -E "LOGJAM.*VULNERABLE.*768" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/LOGJAM_768_CVE-2015-4000.txt
#grep -d skip -E "LOGJAM.*VULNERABLE.*1024" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/LOGJAM_1024_CVE-2015-4000.txt
#grep -d skip -E "BEAST.*SSL3:" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/BEAST_SSL3_CVE-2011-3389.txt
#grep -d skip -E "BEAST.*TLS1:" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/BEAST_TLS1_CVE-2011-3389.txt
#grep -d skip -E "LUCKY13.*VULNERABLE" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/LUCKY13_CVE-2013-0169.txt
#grep -d skip -E "Winshock.*VULNERABLE" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/Winshock_CVE-2014-6321.txt
#grep -d skip -E "RC4.*VULNERABLE" "$PWD/"* | sed -E 's/.*\/([^:]*):.*/\1/; s/_p/:/; s/(:[0-9]*).*/\1/' > parsed_testssl/RC4_CVE-2013-2566_CVE-2015-2808.txt

echo "Cleaning Up..."
echo
find ./parsed_testssl -size 0 -delete
sleep 2
echo "Finished!"