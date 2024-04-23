#date: 2024-04-23T16:52:55Z
#url: https://api.github.com/gists/0ff2f8ad1afc647948566c62a5a43ffa
#owner: https://api.github.com/users/gbrayut

$ echo "GET /" | openssl s_client -showcerts -servername www.linkedin.com -connect www.linkedin.com:443 | openssl x509 -noout -text
depth=2 C = US, O = DigiCert Inc, OU = www.digicert.com, CN = DigiCert Global Root G2
verify return:1
depth=1 C = US, O = Microsoft Corporation, CN = Microsoft Azure RSA TLS Issuing CA 04
verify return:1
depth=0 C = US, ST = WA, L = Redmond, O = Microsoft Corporation, CN = *.azureedge.net
verify return:1
DONE
Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number:
            33:00:2c:88:ce:85:cd:9a:dc:fc:24:fb:8f:00:00:00:2c:88:ce
        Signature Algorithm: sha384WithRSAEncryption
        Issuer: C = US, O = Microsoft Corporation, CN = Microsoft Azure RSA TLS Issuing CA 04
        Validity
            Not Before: Mar 30 00:34:27 2024 GMT
            Not After : Mar 25 00:34:27 2025 GMT
        Subject: C = US, ST = WA, L = Redmond, O = Microsoft Corporation, CN = *.azureedge.net
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                Public-Key: (2048 bit)
                Modulus:
                    00:c1:2c:5d:1c:78:cd:e9:c2:1e:7c:6f:c5:98:08:
                    db:00:2d:7a:8c:5a:2b:28:0c:6f:8d:1e:0c:a6:e1:
                    0f:c9:d9:71:16:2c:92:f2:23:9d:16:19:db:d3:8b:
                    1f:a9:5d:6e:5c:ad:f3:2b:dd:13:ed:8f:3d:d0:dd:
                    4a:ef:d5:e3:fb:f7:5a:08:70:bb:40:d0:43:98:d2:
                    d7:44:a0:77:97:4e:9c:36:35:71:d7:2c:04:1f:de:
                    e4:91:7f:f4:ae:c4:a3:10:03:d0:0a:c0:b9:c5:2f:
                    9f:7a:14:1d:0c:84:75:0b:d2:cb:c8:a4:7a:98:7c:
                    c7:62:5e:4d:ab:00:6e:b8:40:0d:a1:06:47:01:78:
                    ac:91:41:dd:0d:4d:ac:a4:ea:9c:76:1d:94:b3:37:
                    17:ff:8c:bd:e3:20:17:ae:28:a6:dc:d1:50:32:2c:
                    27:d2:79:8e:cd:f2:a8:b5:ad:e7:08:c0:0f:d6:48:
                    55:bc:26:76:10:90:0b:6d:6f:19:57:30:a9:7e:59:
                    8b:aa:a0:4b:09:98:4d:57:ed:72:b4:83:14:6b:c8:
                    19:51:48:32:7a:08:8b:c6:b7:e2:d4:28:d9:91:a5:
                    46:92:54:24:ad:1b:78:38:1d:62:96:35:c3:32:b5:
                    50:85:e5:d4:f8:f0:74:f1:ef:2a:37:42:7f:ab:80:
                    cb:a9
                Exponent: 65537 (0x10001)
        X509v3 extensions:
            CT Precertificate SCTs: 
                Signed Certificate Timestamp:
                    Version   : v1 (0x0)
                    Log ID    : CF:11:56:EE:D5:2E:7C:AF:F3:87:5B:D9:69:2E:9B:E9:
                                1A:71:67:4A:B0:17:EC:AC:01:D2:5B:77:CE:CC:3B:08
                    Timestamp : Mar 30 00:44:30.226 2024 GMT
                    Extensions: none
                    Signature : ecdsa-with-SHA256
                                30:46:02:21:00:85:E4:B2:0D:2F:60:5C:C5:0F:CE:DB:
                                D6:09:E2:3D:E6:62:19:3D:7C:04:E6:02:66:27:FD:EF:
                                0C:BB:5A:9A:83:02:21:00:E5:E7:33:D2:3C:1C:D7:94:
                                0B:E0:B5:0B:1E:4C:49:A5:66:5C:33:21:6A:9E:BC:68:
                                82:BE:C3:79:0D:1C:48:83
                Signed Certificate Timestamp:
                    Version   : v1 (0x0)
                    Log ID    : A2:E3:0A:E4:45:EF:BD:AD:9B:7E:38:ED:47:67:77:53:
                                D7:82:5B:84:94:D7:2B:5E:1B:2C:C4:B9:50:A4:47:E7
                    Timestamp : Mar 30 00:44:30.486 2024 GMT
                    Extensions: none
                    Signature : ecdsa-with-SHA256
                                30:45:02:20:0C:4F:AC:0B:EC:05:C9:D4:5B:A2:53:F0:
                                65:47:F9:BD:0E:75:56:6F:3B:C6:1C:C1:4B:70:19:ED:
                                C9:DD:F2:6B:02:21:00:93:51:7E:D6:38:30:EC:CC:85:
                                AA:3A:CA:28:F0:A7:09:AA:BC:91:4F:15:12:F1:1B:B0:
                                F4:D9:17:37:04:95:AC
                Signed Certificate Timestamp:
                    Version   : v1 (0x0)
                    Log ID    : 4E:75:A3:27:5C:9A:10:C3:38:5B:6C:D4:DF:3F:52:EB:
                                1D:F0:E0:8E:1B:8D:69:C0:B1:FA:64:B1:62:9A:39:DF
                    Timestamp : Mar 30 00:44:30.143 2024 GMT
                    Extensions: none
                    Signature : ecdsa-with-SHA256
                                30:44:02:20:62:F4:B1:13:CE:78:DB:CF:12:55:47:32:
                                78:AA:75:EB:95:D6:81:8E:8F:15:42:18:BC:00:71:60:
                                6B:B8:95:6C:02:20:7A:21:C6:3B:EE:A0:70:A2:3F:C9:
                                6B:15:2A:06:BE:41:40:1D:36:B2:00:B0:DF:AA:92:FB:
                                5A:3B:C5:EF:DE:10
            1.3.6.1.4.1.311.21.10: 
                0.0
..+.......0
..+.......
            1.3.6.1.4.1.311.21.7: 
                0-.%+.....7.........F...........]...i...>..d..&
            Authority Information Access: 
                CA Issuers - URI:http://www.microsoft.com/pkiops/certs/Microsoft%20Azure%20RSA%20TLS%20Issuing%20CA%2004%20-%20xsign.crt
                OCSP - URI:http://oneocsp.microsoft.com/ocsp
            X509v3 Subject Key Identifier: 
                78:03:ED:86:12:F5:78:7C:06:7E:DF:C6:0F:E1:5B:D8:CB:74:32:6B
            X509v3 Key Usage: critical
                Digital Signature, Key Encipherment
            X509v3 Subject Alternative Name: 
                DNS:*.azureedge.net, DNS:*.media.microsoftstream.com, DNS:*.origin.mediaservices.windows.net, DNS:*.streaming.mediaservices.windows.net
            X509v3 Basic Constraints: critical
                CA:FALSE
            X509v3 CRL Distribution Points: 
                Full Name:
                  URI:http://www.microsoft.com/pkiops/crl/Microsoft%20Azure%20RSA%20TLS%20Issuing%20CA%2004.crl
            X509v3 Certificate Policies: 
                Policy: 1.3.6.1.4.1.311.76.509.1.1
                  CPS: http://www.microsoft.com/pkiops/Docs/Repository.htm
                Policy: 2.23.140.1.2.2
            X509v3 Authority Key Identifier: 
                3B:70:D1:53:E9:76:25:9D:60:A8:CA:66:0F:C6:9B:AE:6F:54:16:6A
            X509v3 Extended Key Usage: 
                TLS Web Client Authentication, TLS Web Server Authentication
    Signature Algorithm: sha384WithRSAEncryption
    Signature Value:
        6c:df:a1:d8:db:2a:83:63:4d:ed:fa:9a:ca:0b:d5:e8:84:c9:
        3a:6c:a5:db:4f:29:ca:51:22:62:5b:56:ee:bb:b2:22:29:92:
        8c:0b:f8:ca:7d:d2:ad:11:6b:c7:14:90:a3:c7:36:92:e0:c1:
        e5:3f:a6:a9:8e:ce:88:53:08:b1:26:dc:f6:bb:b5:7c:fb:4b:
        69:e9:d2:5e:e8:a5:c0:4d:ea:f4:fe:d9:a2:51:19:5b:9e:86:
        62:58:6e:05:6e:30:5d:d7:64:8d:99:e3:03:13:6c:9b:4d:48:
        04:c8:fd:84:52:e5:61:f0:2d:e8:b1:f0:2b:7d:7a:ed:a4:b3:
        b7:28:4f:90:dd:ac:e5:ba:ab:9f:b7:3f:fc:9b:ff:18:65:4e:
        70:0d:de:91:75:e9:a2:76:85:e1:47:c1:b6:ff:a5:ef:98:e7:
        b6:4d:0a:6e:7d:bc:fc:26:c6:2f:af:61:40:af:df:c9:71:b7:
        75:fd:59:b8:ef:bf:25:bd:89:54:25:ea:e3:f6:b3:79:e6:81:
        2f:af:a8:0b:88:99:15:76:96:31:43:04:05:0f:ca:c8:39:05:
        1a:95:7a:a5:dd:9c:ed:b1:a8:7d:9e:7e:15:e1:a7:22:93:a2:
        41:7b:35:f0:05:46:95:7c:a2:68:75:f3:d1:b7:91:45:9f:2b:
        87:bc:d6:de:63:e2:eb:0b:71:12:f5:45:8c:3a:eb:f2:0c:40:
        25:4f:67:32:cd:e2:65:b4:b2:4a:65:0b:20:d1:17:da:d6:e2:
        86:d8:f4:fa:92:81:6e:ef:2e:86:ba:6e:bc:13:6e:10:49:10:
        fd:38:c9:ac:38:9a:1f:b8:1b:1a:dd:e4:bf:9a:73:39:ff:73:
        86:14:5b:96:68:90:57:f7:16:ec:d9:58:8f:0e:cd:1b:a6:8d:
        02:e7:34:14:f7:cd:f5:84:1b:5c:cb:f5:8a:08:80:d3:06:68:
        6f:59:49:dd:a0:d9:24:7a:46:3b:1d:bb:a3:8a:6f:14:bf:37:
        a6:8c:ca:15:7a:25:19:61:df:d6:d3:21:46:62:59:ae:d8:67:
        f2:34:6f:27:55:ed:f4:ae:e8:17:3e:1a:56:08:a4:0a:d0:1a:
        cf:cc:bb:2b:4f:8a:56:67:8b:d2:a4:cb:ca:56:fc:c0:e6:ff:
        f0:92:45:77:d1:31:1e:2c:55:1d:d1:ac:0f:a0:ee:2a:cf:a0:
        49:00:67:df:50:c8:a8:53:02:dd:93:bc:6d:ef:23:3b:49:13:
        c2:d6:26:62:c6:64:2d:59:a3:c4:f9:7c:2b:85:ab:16:f0:d3:
        58:4d:fe:32:00:7e:d7:ae:bd:f9:8f:47:2a:b1:8b:f7:74:ff:
        48:d2:94:9c:fe:31:b2:13
