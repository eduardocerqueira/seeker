#date: 2024-05-08T16:59:37Z
#url: https://api.github.com/gists/ee1f54a2044fd1b07172127c8c455a35
#owner: https://api.github.com/users/DarkOperation

# https://wiki.vg/Server_List_Ping#1.4_to_1.5
import socket


def parse_motd(inp):
    motd = ''
    n = False
    for char in inp:
        if char == '§' or n:
            if n:
                n = not n
            else:
                n = True
            pass
        else:
            motd += char
    return motd


class MCInfo:
    def __init__(self, inp):
        null_char, version_id, version, motd, players_online, players_max = inp.split("\x00")
        del null_char
        self.motd = parse_motd(motd)
        self.protocol_number = version_id
        self.mc_version = version
        self.players_online = players_online
        self.players_max = players_max

    def __str__(self):
        return f'Protocol Number: {self.protocol_number} | Minecraft Version: {self.mc_version} | Players: {self.players_online} / {self.players_max} | Motd: {self.motd}'


def status(host, port=25565):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.send(b"\xFE\x01")
    data = s.recv(1024)
    s.close()

    data = data[3:]
    res_str = data.decode("UTF-16-be", errors="ignore")
    print(MCInfo(res_str))


status("localhost")
