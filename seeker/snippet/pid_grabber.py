#date: 2025-12-03T17:12:25Z
#url: https://api.github.com/gists/65dc33056278d0446a51770d3907c0bd
#owner: https://api.github.com/users/spoongaming61

from tcpgecko import TCPGecko

gecko = TCPGecko("192.168.1.1")  # Your Wii U's LAN IP address goes here

player_ptr = int.from_bytes(
    gecko.readmem(int.from_bytes(gecko.readmem(0x106E0330, 4), "big") + 0x10, 4), "big"
)  # Load pointer-in-pointer

for offset in range(0x0, 0x1C + 1, 0x4):
    player = int.from_bytes(gecko.readmem(player_ptr + offset, 4), "big")
    player_name = (
        ((gecko.readmem(player + 0x6, 20)).decode("utf-16-be")).strip()
    ).replace("\u003F", "")
    player_pid = int.from_bytes(gecko.readmem(player + 0xD0, 4), "big")

    print(
        ("Player {0} | PID (Hex): {1} | PID (Dec): {2} | Mii name: {3}").format(
            offset // 0x4 + 1, hex(player_pid), player_pid, player_name
        )
    )