#date: 2023-03-15T16:56:35Z
#url: https://api.github.com/gists/51d30a2cd243d6f86591aef42af63b4a
#owner: https://api.github.com/users/SimonTheCoder

import OpenSSL.crypto as crypto
import sys
# 读取DER格式的X.509数字证书文件
with open(sys.argv[1], 'rb') as cert_file:
    cert_data = cert_file.read()

scan_count = 0

while True:
    try:
        print(f"trying offset:{scan_count}",)
        try_cert = cert_data[scan_count:]
        # 解析X.509数字证书
        x509_cert = crypto.load_certificate(crypto.FILETYPE_ASN1, try_cert)
        # 读取证书信息
        subject = x509_cert.get_subject()
        issuer = x509_cert.get_issuer()
        serial_number = x509_cert.get_serial_number()
        not_before = x509_cert.get_notBefore()
        not_after = x509_cert.get_notAfter()
        public_key = x509_cert.get_pubkey()

        # 输出证书信息
        print('Subject:', subject)
        print('Issuer:', issuer)
        print('Serial number:', serial_number)
        print('Not before:', not_before)
        print('Not after:', not_after)
        print('Public key:', public_key)

        cert_binary_data = crypto.dump_certificate(crypto.FILETYPE_ASN1, x509_cert)
        cert_size = len(cert_binary_data)
        with open(f"cert@{scan_count}.der","wb") as f:
            f.write(cert_binary_data)
        scan_count=scan_count + cert_size
    except Exception as e:
        #print(e)
        #print("Failed.")
        scan_count = scan_count + 1

# # 读取证书信息
# subject = x509_cert.get_subject()
# issuer = x509_cert.get_issuer()
# serial_number = x509_cert.get_serial_number()
# not_before = x509_cert.get_notBefore()
# not_after = x509_cert.get_notAfter()
# public_key = x509_cert.get_pubkey()

# # 输出证书信息
# print('Subject:', subject)
# print('Issuer:', issuer)
# print('Serial number:', serial_number)
# print('Not before:', not_before)
# print('Not after:', not_after)
# print('Public key:', public_key)
