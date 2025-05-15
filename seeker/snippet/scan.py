#date: 2025-05-15T16:45:31Z
#url: https://api.github.com/gists/38e219305e477c36e9463b468202c8ee
#owner: https://api.github.com/users/wku

import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Set

from web3 import Web3
from web3.middleware import geth_poa_middleware
from web3.exceptions import BlockNotFound, TransactionNotFound

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('polygon_scanner')

ERC20_TRANSFER_EVENT = Web3.keccak(text="Transfer(address,address,uint256)").hex()
ERC721_TRANSFER_EVENT = Web3.keccak(text="Transfer(address,address,uint256)").hex()
ERC1155_TRANSFER_SINGLE_EVENT = Web3.keccak(text="TransferSingle(address,address,address,uint256,uint256)").hex()
ERC1155_TRANSFER_BATCH_EVENT = Web3.keccak(text="TransferBatch(address,address,address,uint256[],uint256[])").hex()


 "**********"c "**********"l "**********"a "**********"s "**********"s "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********"T "**********"r "**********"a "**********"n "**********"s "**********"f "**********"e "**********"r "**********"E "**********"v "**********"e "**********"n "**********"t "**********": "**********"

    def __init__(self, tx_hash: "**********": int, token_address: str,
                 token_type: "**********": str, to_address: str,
                 token_id: "**********": Optional[int] = None,
                 token_symbol: "**********": int = 18):
        self.tx_hash = tx_hash
        self.block_number = block_number
        self.token_address = "**********"
        self.token_type = "**********"
        self.from_address = from_address
        self.to_address = to_address
        self.token_id = "**********"
        self.value = value
        self.token_symbol = "**********"
        self.token_decimals = "**********"
        self.timestamp = int(time.time())

    def get_formatted_value(self) -> str:
        if self.value is None:
            return "N/A"

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"t "**********"y "**********"p "**********"e "**********"  "**********"= "**********"= "**********"  "**********"' "**********"E "**********"R "**********"C "**********"2 "**********"0 "**********"' "**********": "**********"
            formatted_value = "**********"
            return f"{formatted_value:.8f}".rstrip('0').rstrip('.') if '.' in f"{formatted_value:.8f}" else f"{formatted_value:.8f}"
        else:
            return str(self.value)

    def __str__(self):
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"t "**********"y "**********"p "**********"e "**********"  "**********"= "**********"= "**********"  "**********"' "**********"E "**********"R "**********"C "**********"2 "**********"0 "**********"' "**********": "**********"
            return (f"[{self.token_type}] Block: "**********"
                    f"Tx: "**********": {self.token_symbol} ({self.token_address}) | "
                    f"From: {self.from_address} | To: {self.to_address} | "
                    f"Value: "**********"
        else:
            return (f"[{self.token_type}] Block: "**********"
                    f"Tx: "**********": {self.token_symbol} ({self.token_address}) | "
                    f"From: {self.from_address} | To: {self.to_address} | "
                    f"TokenID: "**********": {self.get_formatted_value()}")


class PolygonScanner:

    def __init__(self, wallet_addresses: List[str], start_block: Optional[int] = None, scan_interval: int = 30):
        self.wallet_addresses = [addr.lower() for addr in wallet_addresses]
        self.wallet_set = set(self.wallet_addresses)
        self.scan_interval = scan_interval
        self.token_info_cache = "**********"

        ankr_rpc_url = "https://rpc.ankr.com/polygon/fb00ba52a7de8c2eb4acca0df5590553673491333704db7739fbdf8a40d0f1ad"
        self.w3 = Web3(Web3.HTTPProvider(ankr_rpc_url))
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)

        if not self.w3.is_connected():
            raise ConnectionError("Не удалось подключиться к блокчейну Polygon")

        logger.info(f"Успешное подключение к Polygon. Chain ID: {self.w3.eth.chain_id}")

        self.last_processed_block = start_block or self.w3.eth.block_number - 10
        logger.info(f"Начало сканирования с блока {self.last_processed_block}")

    async def scan_blocks_loop(self):
        while True:
            try:
                current_block = self.w3.eth.block_number
                if current_block > self.last_processed_block:
                    logger.info(f"Сканирование блоков с {self.last_processed_block + 1} по {current_block}")
                    await self.process_blocks(self.last_processed_block + 1, current_block)
                    self.last_processed_block = current_block
                else:
                    logger.info(f"Ожидание новых блоков. Текущий блок: {current_block}")

                await asyncio.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"Ошибка в цикле сканирования: {str(e)}")
                await asyncio.sleep(5)

    async def process_blocks(self, from_block: int, to_block: int):
        tasks = [self.process_block(block_num) for block_num in range(from_block, to_block + 1)]
        await asyncio.gather(*tasks)

    async def process_block(self, block_number: int):
        try:
            block = await asyncio.to_thread(self.w3.eth.get_block, block_number, full_transactions=True)

            for tx in block.transactions:
                await self.process_transaction(tx, block_number)

        except BlockNotFound:
            logger.warning(f"Блок {block_number} не найден")
        except Exception as e:
            logger.error(f"Ошибка при обработке блока {block_number}: {str(e)}")

    async def process_transaction(self, tx, block_number: int):
        tx_hash = tx['hash'].hex()
        try:
            from_address = tx['from'].lower()
            to_address = tx.get('to', '').lower() if tx.get('to') else None

            if from_address in self.wallet_set or to_address in self.wallet_set:
                try:
                    receipt = await asyncio.to_thread(self.w3.eth.get_transaction_receipt, tx_hash)
                except TransactionNotFound:
                    logger.debug(f"Транзакция {tx_hash} не найдена, пропускаем")
                    return

                if not receipt or not hasattr(receipt, 'logs'):
                    logger.debug(f"Отсутствуют логи для транзакции {tx_hash}")
                    return

                for log in receipt.logs:
                    contract_address = log['address'].lower()
                    topics = [t.hex() if isinstance(t, bytes) else t for t in log['topics']]

                    if not topics:
                        continue

                    event_signature = topics[0]

                    if event_signature == ERC20_TRANSFER_EVENT:
                        if len(topics) >= 3:
                            from_addr = '0x' + topics[1][26:].lower()
                            to_addr = '0x' + topics[2][26:].lower()

                            if from_addr in self.wallet_set or to_addr in self.wallet_set:
                                token_type = "**********"
                                token_symbol, token_decimals = "**********"

                                data_str = self._normalize_data(log['data'])

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"t "**********"y "**********"p "**********"e "**********"  "**********"= "**********"= "**********"  "**********"' "**********"E "**********"R "**********"C "**********"7 "**********"2 "**********"1 "**********"' "**********": "**********"
                                    try:
                                        token_id = "**********"= '0x' else None
                                    except ValueError:
                                        logger.warning(f"Невозможно преобразовать данные в токен ID: {data_str}")
                                        token_id = "**********"

                                    event = "**********"
                                        tx_hash=tx_hash,
                                        block_number=block_number,
                                        token_address= "**********"
                                        token_type= "**********"
                                        from_address=from_addr,
                                        to_address=to_addr,
                                        token_id= "**********"
                                        value=1,
                                        token_symbol= "**********"
                                        token_decimals= "**********"
                                    )
                                else:
                                    try:
                                        value = int(data_str, 16) if data_str != '0x' else 0
                                    except ValueError:
                                        logger.warning(f"Невозможно преобразовать данные в значение: {data_str}")
                                        value = 0

                                    event = "**********"
                                        tx_hash=tx_hash,
                                        block_number=block_number,
                                        token_address= "**********"
                                        token_type= "**********"
                                        from_address=from_addr,
                                        to_address=to_addr,
                                        value=value,
                                        token_symbol= "**********"
                                         "**********"= "**********"
                                    )

                                print(event)

                    elif event_signature == ERC1155_TRANSFER_SINGLE_EVENT:
                        if len(topics) >= 4:
                            from_addr = '0x' + topics[2][26:].lower()
                            to_addr = '0x' + topics[3][26:].lower()

                            if from_addr in self.wallet_set or to_addr in self.wallet_set:
                                token_symbol, token_decimals = "**********"
                                data_str = self._normalize_data(log['data'])
                                if data_str.startswith('0x'):
                                    data_str = data_str[2:]

                                if len(data_str) >= 128:
                                    try:
                                        token_id = int(data_str[: "**********"
                                        value = int(data_str[64:128], 16)
                                    except ValueError as e:
                                        logger.warning(f"Ошибка при преобразовании данных ERC1155: {str(e)}")
                                        token_id = "**********"
                                        value = 0

                                    event = "**********"
                                        tx_hash=tx_hash,
                                        block_number=block_number,
                                        token_address= "**********"
                                        token_type= "**********"
                                        from_address=from_addr,
                                        to_address=to_addr,
                                        token_id= "**********"
                                        value=value,
                                        token_symbol= "**********"
                                        token_decimals= "**********"
                                    )
                                    print(event)

                    elif event_signature == ERC1155_TRANSFER_BATCH_EVENT:
                        if len(topics) >= 4:
                            from_addr = '0x' + topics[2][26:].lower()
                            to_addr = '0x' + topics[3][26:].lower()

                            if from_addr in self.wallet_set or to_addr in self.wallet_set:
                                token_symbol, token_decimals = "**********"
                                event = "**********"
                                    tx_hash=tx_hash,
                                    block_number=block_number,
                                    token_address= "**********"
                                    token_type= "**********"
                                    from_address=from_addr,
                                    to_address=to_addr,
                                    token_id= "**********"
                                    value=None,
                                    token_symbol= "**********"
                                    token_decimals= "**********"
                                )
                                print(f"{event} (Batch transfer - подробности в транзакции)")

        except TransactionNotFound:
            logger.debug(f"Транзакция {tx_hash} не найдена при обработке")
        except Exception as e:
            logger.error(f"Ошибка при обработке транзакции {tx_hash}: {str(e)}")

    def _normalize_data(self, data) -> str:
        if isinstance(data, bytes):
            data_hex = data.hex()
            if not data_hex.startswith('0x'):
                data_hex = '0x' + data_hex
            return data_hex
        elif isinstance(data, str):
            if not data.startswith('0x'):
                return '0x' + data
            return data
        else:
            return '0x' + str(data)

    async def _determine_token_type(self, contract_address: "**********":
        try:
            checksum_address = Web3.to_checksum_address(contract_address)
            abi = [{"constant": True, "inputs": [{"name": "interfaceId", "type": "bytes4"}],
                    "name": "supportsInterface", "outputs": [{"name": "", "type": "bool"}],
                    "payable": False, "stateMutability": "view", "type": "function"}]

            contract = self.w3.eth.contract(address=checksum_address, abi=abi)

            try:
                supports_erc721 = await asyncio.to_thread(
                    contract.functions.supportsInterface(Web3.to_bytes(hexstr='0x80ac58cd')).call
                )
                if supports_erc721:
                    return 'ERC721'

                supports_erc1155 = await asyncio.to_thread(
                    contract.functions.supportsInterface(Web3.to_bytes(hexstr='0xd9b67a26')).call
                )
                if supports_erc1155:
                    return 'ERC1155'
            except Exception:
                pass

            erc20_abi = [
                {"constant": True, "inputs": [], "name": "decimals",
                 "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
                {"constant": True, "inputs": [], "name": "symbol",
                 "outputs": [{"name": "", "type": "string"}], "type": "function"}
            ]

            erc20_contract = self.w3.eth.contract(address=checksum_address, abi=erc20_abi)
            try:
                await asyncio.to_thread(erc20_contract.functions.decimals().call) or \
                await asyncio.to_thread(erc20_contract.functions.symbol().call)
                return 'ERC20'
            except Exception:
                pass

            return 'ERC20'
        except Exception as e:
            logger.debug(f"Ошибка определения типа токена {contract_address}: {str(e)}")
            return 'ERC20'

    async def _get_token_info(self, contract_address: "**********": str) -> (str, int):
        cache_key = contract_address.lower()
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"c "**********"a "**********"c "**********"h "**********"e "**********"_ "**********"k "**********"e "**********"y "**********"  "**********"i "**********"n "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"n "**********"f "**********"o "**********"_ "**********"c "**********"a "**********"c "**********"h "**********"e "**********": "**********"
            return self.token_info_cache[cache_key]

        symbol = "UNKNOWN"
        decimals = 18

        try:
            checksum_address = Web3.to_checksum_address(contract_address)

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"t "**********"y "**********"p "**********"e "**********"  "**********"= "**********"= "**********"  "**********"' "**********"E "**********"R "**********"C "**********"2 "**********"0 "**********"' "**********": "**********"
                erc20_abi = [
                    {"constant": True, "inputs": [], "name": "decimals",
                     "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
                    {"constant": True, "inputs": [], "name": "symbol",
                     "outputs": [{"name": "", "type": "string"}], "type": "function"}
                ]

                contract = self.w3.eth.contract(address=checksum_address, abi=erc20_abi)

                try:
                    symbol = await asyncio.to_thread(contract.functions.symbol().call)
                except Exception as e:
                    logger.debug(f"Не удалось получить символ токена {contract_address}: {str(e)}")

                try:
                    decimals = await asyncio.to_thread(contract.functions.decimals().call)
                except Exception as e:
                    logger.debug(f"Не удалось получить decimals токена {contract_address}: {str(e)}")

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"e "**********"l "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"t "**********"y "**********"p "**********"e "**********"  "**********"i "**********"n "**********"  "**********"[ "**********"' "**********"E "**********"R "**********"C "**********"7 "**********"2 "**********"1 "**********"' "**********", "**********"  "**********"' "**********"E "**********"R "**********"C "**********"1 "**********"1 "**********"5 "**********"5 "**********"' "**********"] "**********": "**********"
                nft_abi = [
                    {"constant": True, "inputs": [], "name": "symbol",
                     "outputs": [{"name": "", "type": "string"}], "type": "function"},
                    {"constant": True, "inputs": [], "name": "name",
                     "outputs": [{"name": "", "type": "string"}], "type": "function"}
                ]

                contract = self.w3.eth.contract(address=checksum_address, abi=nft_abi)

                try:
                    symbol = await asyncio.to_thread(contract.functions.symbol().call)
                except Exception:
                    try:
                        symbol = await asyncio.to_thread(contract.functions.name().call)
                    except Exception as e:
                        logger.debug(f"Не удалось получить имя/символ NFT {contract_address}: {str(e)}")
                        symbol = f"{token_type}-{contract_address[-6: "**********"

                decimals = 0

            self.token_info_cache[cache_key] = "**********"
            return symbol, decimals

        except Exception as e:
            logger.warning(f"Ошибка при получении информации о токене {contract_address}: {str(e)}")
            self.token_info_cache[cache_key] = "**********"
            return symbol, decimals


async def main():
    wallets = [
        '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
        '0x123456789abcdef123456789abcdef123456789a',
        '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff',
        '0x8dFf5E27EA6b7A60262716b3c4f4b81c8c4aB2b9',
        '0x2953399124F0cBB46d2CbACD8A89cF0599974963',
        '0x40ec5B33f54e0E8A33A975908C5BA1c14e5BbbDf',
        '0xc2132D05D31c914a87C6611C10748AEb04B58e8F',
    ]

    scanner = PolygonScanner(wallets)

    logger.info("Запуск сканера блокчейна Polygon")
    await scanner.scan_blocks_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Сканирование остановлено пользователем")
    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}")