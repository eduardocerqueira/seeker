#date: 2026-01-05T17:17:58Z
#url: https://api.github.com/gists/f1d916bbc7d9e9f971e84de35e8ff45c
#owner: https://api.github.com/users/johan-boule

#! /usr/bin/env python3

import sys, socket, struct, dataclasses

host = sys.argv[1] if len(sys.argv) > 1 else 'lobby.wz2100.net'
port = int(sys.argv[2]) if len(sys.argv) > 2 else 9990

@dataclasses.dataclass
class Game:
    binary_format = '!I64si4B40s2i4I40s40s157sH40s40s64s255s9I'

    version: int # ui32
    name: str # 64
    size: int # i32
    alliances: int # ui8
    tech_level: int # ui8
    power_level: int # ui8
    bases_level: int # ui8
    host: str # 40
    max_players: int # i32
    current_players: int # i32
    game_type: int # ui32
    open_spectator_slots: int # ui32
    blind_mode: int # ui32
    unused1: int # ui32
    host2: str # 40
    host3: str # 40
    unused2: str # 157
    host_port: int # ui16
    map_name: str # 40
    host_name: str # 40
    version_string: str # 64
    mod_list: str # 255
    version_major: int # ui32
    version_minor: int # ui32
    private_game: int # ui32
    pure_map: int # ui32
    mods: int # ui32
    game_id: int # ui32
    limits: int # ui32
    unused3: int # ui32
    unused4: int # ui32

    def __post_init__(self):
        def decode(binary):
            return binary.split(b'\0', 1)[0].decode()
        self.name = decode(self.name)
        self.host = decode(self.host)
        self.host2 = decode(self.host2)
        self.host3 = decode(self.host3)
        self.unused2 = decode(self.unused2)
        self.map_name = decode(self.map_name)
        self.host_name = decode(self.host_name)
        self.version_string = decode(self.version_string)
        self.mod_list = decode(self.mod_list)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((host, port))

    def send(binary_format, *data):
        s.sendall(struct.pack(binary_format, *data))

    def receive(binary_format):
        binary_format = struct.Struct(binary_format)
        received = bytearray()
        size_left = binary_format.size
        while size_left > 0:
            received += s.recv(size_left)
            size_left = binary_format.size - len(received)
        return binary_format.unpack(received)

    send('!5s', b'list\n')
    
    count, = receive('!I')
    games = []
    for _ in range(count): games.append(Game(*receive(Game.binary_format)))

    status, = receive('!I')
    print(f'status: {status}')

    print()

    print('motd:')
    motd_len, = receive('!I')
    motd, = receive(f'!{motd_len}s')
    print('\t' + '\n\t'.join(motd.decode().split('\n')))

    print()

    more_games, = receive('!I')
    if more_games & 1 == 1: # See explanation in warzone2100/lib/netplay/netplay.cpp function NETenumerateGames
        count, = receive('!I')
        games = []
        for _ in range(count): games.append(Game(*receive(Game.binary_format)))

    print(f'game count: {count}')

    print()

    for game in games:
        print(f'{game.game_id}\t{game.current_players}/{game.max_players}\t{game.host_name}, {game.map_name}, {game.name}')

    print()

    for game in games: print(game)