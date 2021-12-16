#date: 2021-12-16T17:03:20Z
#url: https://api.github.com/gists/9cc2372c3b12adbbb695352ec5633cdd
#owner: https://api.github.com/users/spc789

from collections import deque
from binance.websockets import BinanceSocketManager
from binance.client import Client
from binance.enums import *
from twisted.internet import reactor

coins = ['ACM', "ADX", "AERGO", "AGI", "AION", "AKRO", "ALPHA", "AMB", "APPC", "ARDR", "ARK", "ASR", "AST", "ATM", 'AUCTION', "AUDIO", "AVA", "AXS", "BADGER", "BCD", "BEAM", "BEL", "BLZ", "BQX", "BRD", "BTCST", "BTG", "BTS", "BZRX", "CAKE", "CDT", "CELO", "CKB", "CND", "COS", "CTK", "CTXC", "DATA", "DCR", "DEGO", "DIA", "DLT", "DNT", "DODO", "DREP", "DUSK", "EASY", "EGLD", "ELF", "EVX", "FIO", "FIRO", "FIS", "FLM", "FOR", "FRONT", "FUN", "FXS", "GAS", "GLM", "GO", "GRS", "GTO", "GVT", "HARD", "HNT", "IDEX", "INJ", "JUV", "KSM", "LIT", "LOOM", "LTO", "LUNA", "MDA", "MDT", "MITH", "MTH", "MTL", "NAS", "NAV", "NBS", "NEBL", "NKN", "NMR", "NXS", "OAX", "OCEAN", "OG", "ORN", "OST", "OXT", "PAXG", "PERL", "PHB", "PIVX", "PNT", "POA", "POLY", "POWR", "PPT", "PSG", "QKC", "QLC", "QSP", "RCN", "RDN", "REEF", "RENBTC", "REQ", "RIF", "ROSE", "RUNE", "SCRT", "SKL", "SKY", "SNGLS", "SNM", "SNT", "STEEM", "STMX", "STPT", "STRAX", "STX", "SUN", "SUSD", "SYS", "TNB", "TRU", "TWT", "UMA", "UNFI", "UTK", "VIA", "VIB", "VIBE", "VIDT", "VITE", "WABI", "WAN", "WBTC", "WING", "WPR", "XEM", "XVG", "XVS", "XZC", "YOYO", "ZEN"]

MAX_TRANS = 500

class CryptoPumpDetector():
    def __init__(self) -> None:
        self.client = Client('BINANCE_API_KEY', 'BINANCE_SECRET_KEY')
        self.queues = {f'{coin}BTC': deque(maxlen=MAX_TRANS) for coin in coins}
        self.bm = BinanceSocketManager(self.client)
        self.conns = []

    def start(self):
        for c in shitcoins:
            self.conns.append(self.bm.start_aggtrade_socket(f'{c}BTC', self.__process_crypto_events))

        self.bm.start()
    
    def __process_crypto_events(self, e):
        symbol, price, eTime = e['s'], float(e['p']), int(e['E'])
        q = self.queues[symbol]

        if len(q) > 0:
            pct = price / q[0]
            if pct >= 1.1: # price increase by 10%
                q.clear()
                print(f'PUMP {symbol} !!!!!')
        q.append(price)

    def close(self):
        for con in self.conns:
            self.bm.stop_socket(con)
        reactor.stop()
        
if __name__ == '__main__':
    pump_det = CryptoPumpDetector()
    pump_det.start(context.bot)
    
    try:
        while True:
            pass
    except KeyboardInterrupt:
      pump_det.close()