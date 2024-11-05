#date: 2024-11-05T17:07:28Z
#url: https://api.github.com/gists/7361eb541c4b08960c47e58b7bb463e1
#owner: https://api.github.com/users/sourceperl

import time
from collections import defaultdict
from typing import Dict

from pyModbusTCP.server import DataBank, ModbusServer


class MyDataBank(DataBank):
    class Data:
        def __init__(self) -> None:
            self.h_regs = {}

    def __init__(self):
        # turn off allocation of memory for standard modbus object types
        # only "holding registers" space will be replaced by dynamic build values.
        super().__init__(virtual_mode=True)
        # public
        self.uids: Dict[int, MyDataBank.Data] = defaultdict(MyDataBank.Data)

    def get_holding_registers(self, address: int, number: int, srv_info: ModbusServer.ServerInfo):
        # returns None if any of the registers are missing
        try:
            req_uid = srv_info.recv_frame.mbap.unit_id
            return [self.uids[req_uid].h_regs[addr] for addr in range(address, address+number)]
        except KeyError:
            return


if __name__ == '__main__':
    # init a custom data bank, fill it with static data for unit id #1
    data_bank = MyDataBank()
    data_bank.uids[1].h_regs[0] = 42
    # init modbus server and start it
    ModbusServer(host='localhost', port=5020, data_bank=data_bank, no_block=True).start()
    # update dynamic data at @0 for unit id #2
    while True:
        for i in range(1_000):
            data_bank.uids[2].h_regs[0] = i
            time.sleep(0.1)
