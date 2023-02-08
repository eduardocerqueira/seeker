#date: 2023-02-08T16:55:21Z
#url: https://api.github.com/gists/2c9b763b510828425528b393c6ff483a
#owner: https://api.github.com/users/Synceratus

import re


def convert_address(binary_address):
    decimal_address = [binary_address[0:8], binary_address[8:16], binary_address[16:24], binary_address[24:32]]
    decimal_address = ".".join(str(int(byte, 2)) for byte in decimal_address)
    return decimal_address


def get_protocol(protocol_numer):
    return {
        '6': "TCP",
        '17': "UDP"
    }.get(str(protocol_numer), protocol_numer)


def beautify_header(header):
    header['Version'] = "IPv4" if int(header['Version'], 2) == 4 else "Error"
    header['IHL'] = str(int(header['IHL'], 2)) + " bytes"
    header['DSCP'] = "0b" + str(header['DSCP'])
    header['ECN'] = "0b" + str(header['ECN'])
    header['Total Length'] = str(int(header['Total Length'], 2)) + " bytes"
    header['Identification'] = "0b" + str(header['Identification'])
    header['Flags'] = "0b" + str(header['Flags'])
    header['Fragment Offset'] = "0b" + str(header['Fragment Offset'])
    header['TTL'] = str(int(header['TTL'], 2)) + " seconds"
    header['Protocol'] = get_protocol(header['Protocol'])
    header['Header Checksum'] = "0b" + str(header['Header Checksum'])
    header['Source IP Address'] = convert_address(header['Source IP Address'])
    header['Destination IP Address'] = convert_address(header['Destination IP Address'])
    header['Options'] = "N/A" if header['Options'] == "" else "0b" + str(header['Options'])
    return header


def disassemble_header(data):
    if re.match(r'^[0-1]+$', data) is not None:
        header = {
            'Version': data[:4],
            'IHL': data[4:8],
            'DSCP': data[8:14],
            'ECN': data[14:16],
            'Total Length': data[16:32],
            'Identification': data[32:48],
            'Flags': "0b" + data[48:51],
            'Fragment Offset': data[51:64],
            'TTL': data[64:72],
            'Protocol': data[72:80],
            'Header Checksum': data[80:96],
            'Source IP Address': data[96:128],
            'Destination IP Address': data[128:160],
            'Options': ""
        }
        if int(header['IHL'], 2) > 5:
            header['Options'] = data[160:(int(header['IHL'], 2) - 5) * 32]
        return beautify_header(header)
    else:
        print("Error, provided data is not in binary format.")


if __name__ == '__main__':
    hex_data = input("Provide your IPv4 packet in hex format:").replace(" ", "")
    binary_data = bin(int(hex_data, 16))[2:].zfill(len(hex_data * 4))
    ip_header = disassemble_header(binary_data)

    for key, value in ip_header.items():
        print(f"{key}: {value}")
