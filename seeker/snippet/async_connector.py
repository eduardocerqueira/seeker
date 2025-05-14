#date: 2025-05-14T17:09:11Z
#url: https://api.github.com/gists/ced2a8f08b301872e3e205e05e7e8251
#owner: https://api.github.com/users/rafa-be

class AsyncConnector:
    def __init__(context: IOContext, host: str, port: int):
        self._base_connector = context.open_tcp_connection(
            host, port, self.__on_receive_message, self.__on_close
        )
        self._receiving_queue: Queue[asyncio.Future] = Queue()

    # these are called from the Python asyncio loop

    async def send(self, message: Message):
        self.__base_connector.send(message)

    async def receive(self) -> Message:
        receiving_future = asyncio.Future()
        self._receiving_queue.add(receiving_future)
        await receiving_future

    # these are called from the Python thread running the C++ IO context

    def __on_receive_message(self, message: Message):
        receiving_future = self._receiving_queue.pop()
        receiving_future.get_loop().call_soon_threadsafe(receiving_future.set_result, message)

    def __on_close(self):
        ...
