#date: 2025-06-02T17:10:16Z
#url: https://api.github.com/gists/c243e7441548a6bc9281d0cec1e68b59
#owner: https://api.github.com/users/teaishealthy

import asyncio
import json
import traceback
import uuid
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Literal,
    Protocol,
    TypedDict,
    TypeVar,
    cast,
)

import redis.asyncio as aioredis

type Flexible = str | int | float | bool | None | dict[str, Flexible] | list[Flexible]


R = TypeVar("R")
type Ack = Callable[[], Awaitable[None]]

type CoroT[**P, R] = Callable[P, Awaitable[R]]
type MaybeCoroT[**P, R] = CoroT[P, R] | Callable[P, R]

class RPCCall(TypedDict):
    type: Literal["call"]
    method: str
    args: list[Flexible]
    kwargs: dict[str, Flexible]


class RPCError(TypedDict):
    type: Literal["error"]
    error: dict[str, int | str]  # Code and message


class RPCResult(TypedDict):
    type: Literal["result"]
    result: Flexible


class RPCAcknowledged(TypedDict):
    type: Literal["acknowledged"]


class RPCIntrospectRequest(TypedDict):
    type: Literal["introspect"]


class RPCIntrospectResponse(TypedDict):
    type: Literal["introspect-response"]
    methods: list[str]


class RPCMessageCall(TypedDict):
    id: str
    payload: RPCCall

class RPCMessageIntrospectRequest(TypedDict):
    id: str
    payload: RPCIntrospectRequest

class RPCMessageResult(TypedDict):
    id: str
    payload: RPCResult

class RPCMessageError(TypedDict):
    id: str
    payload: RPCError

class RPCMessageAcknowledged(TypedDict):
    id: str
    payload: RPCAcknowledged

class RPCMessageIntrospectResponse(TypedDict):
    id: str
    payload: RPCIntrospectResponse

type RPCMessage = (
    RPCMessageCall
    | RPCMessageIntrospectRequest
    | RPCMessageResult
    | RPCMessageError
    | RPCMessageAcknowledged
    | RPCMessageIntrospectResponse
)

class Callback(Protocol):
    def __call__(self, ack: Ack, *args: Any) -> Awaitable[Flexible]: ...

# Client/server message filters for typing purposes
RPCClientMessage = RPCMessageCall  | RPCMessageIntrospectRequest
RPCServerMessage = RPCMessageError | RPCMessageResult | RPCMessageAcknowledged | RPCMessageIntrospectResponse

async def maybe_coro(x: Coroutine[R, Any, Any] | R) -> R:
    """If x is a coroutine, await it; otherwise, return x."""
    if asyncio.iscoroutine(x):
        return await x
    return x

class RPCClient:
    def __init__(self, redis_url: str = "redis://localhost", *, logging: bool =False, ack_timeout: float = 1.0, total_timeout: float =5.0):
        self.redis_url = redis_url
        self.ack_timeout = ack_timeout
        self.total_timeout = total_timeout
        self.logging = logging

        self.type_listeners: dict[str, Callable[[RPCMessage], Awaitable[None]]] = {}
        self.method_listeners: dict[str, Callable[[RPCMessage], Awaitable[None]]] = {}
        self.callback_listeners: dict[str, Callable[[RPCMessage], Awaitable[None]]] = {}

        self.type_listeners["introspect"] = self._introspect_request_handler  # type: ignore
        self.type_listeners["call"] = self._call_request_handler  # type: ignore
        for t in ["acknowledged", "result", "error", "introspect-response"]:
            self.type_listeners[t] = self._response_handler


    async def connect(self):
        self.producer = await aioredis.from_url(self.redis_url)
        self.consumer = await aioredis.from_url(self.redis_url)

        pubsub = self.consumer.pubsub()
        await pubsub.subscribe("teaRPC")

        async def reader():
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                parsed = cast(RPCClientMessage, json.loads(message["data"]))
                handler = self.type_listeners.get(parsed["payload"]["type"])
                if handler:
                    await handler(parsed)

        asyncio.create_task(reader())

    async def dispose(self):
        await self.producer.close()
        await self.consumer.close()

    async def _response_handler(self, message: RPCMessage):
        cb = self.callback_listeners.get(message["id"])
        if cb:
            await maybe_coro(cb(message))

    async def _introspect_request_handler(self, request: RPCMessageIntrospectRequest):
        response: RPCMessageIntrospectResponse = {
            "id": request["id"],
            "payload": {
                "type": "introspect-response",
                "methods": list(self.method_listeners.keys()),
            },
        }
        await self.producer.publish("teaRPC", json.dumps(response))

    async def _call_request_handler(self, request: RPCMessageCall):
        method = request["payload"]["method"]
        listener = self.method_listeners.get(method)
        if listener:
            await maybe_coro(listener(request))

    async def safe_call(self, method: str, *args: Flexible) -> dict[str, Any]:
        id_ = uuid.uuid4().hex
        request: RPCMessageCall = {
            "id": id_,
            "payload": {
                "type": "call",
                "method": method,
                "args": list(args),
                "kwargs": {},
            },
        }

        fut = asyncio.get_event_loop().create_future()

        acknowledged = False

        def timeout_handler():
            if not acknowledged:
                self.callback_listeners.pop(id_, None)
                fut.set_result({
                    "success": False,
                    "error": {"message": "acknowledgement timeout", "code": -1},
                })

        def total_timeout_handler():
            self.callback_listeners.pop(id_, None)
            if not fut.done():
                fut.set_result({
                    "success": False,
                    "error": {"message": "total timeout", "code": -2},
                })

        async def cb(msg: RPCMessage):
            nonlocal acknowledged
            t = msg["payload"]["type"]
            if t == "acknowledged":
                acknowledged = True
            elif t == "result":
                msg = cast(RPCMessageResult, msg)
                self.callback_listeners.pop(id_, None)
                fut.set_result({"success": True, "result": msg["payload"]["result"]})
            elif t == "error":
                msg = cast(RPCMessageError, msg)
                self.callback_listeners.pop(id_, None)
                fut.set_result({"success": False, "error": msg["payload"]["error"]})

        self.callback_listeners[id_] = cb
        await self.producer.publish("teaRPC", json.dumps(request))

        asyncio.get_event_loop().call_later(self.ack_timeout, timeout_handler)
        asyncio.get_event_loop().call_later(self.total_timeout, total_timeout_handler)

        return await fut

    async def call(self, method: str, *args: Any) -> Any:
        result = await self.safe_call(method, *args)
        if not result["success"]:
            raise Exception(result["error"]["message"])
        return result["result"]

    # decorator
    def define(self, method: str) -> Callable[[Callback], Callback]:
        def decorator(fn: Callback) -> Callback:
            if not asyncio.iscoroutinefunction(fn):
                raise ValueError(f"Function {fn.__qualname__} must be a coroutine function")

            self._define(method, fn)
            return fn

        return decorator

    def _define(self, method: str, fn: Callback):
        async def handler(request: RPCMessage):
            async def ack():
                await self.producer.publish("teaRPC", json.dumps({
                    "id": request["id"],
                    "payload": {"type": "acknowledged"},
                }))

            try:
                result = await fn(ack, *request["payload"].get("args", []))
                response: RPCMessageResult = {
                    "id": request["id"],
                    "payload": {"type": "result", "result": result},
                }
                await self.producer.publish("teaRPC", json.dumps(response))
            except Exception:
                error: RPCMessageError = {
                    "id": request["id"],
                    "payload": {
                        "type": "error",
                        "error": {"message": traceback.format_exc(), "code": -1},
                    },
                }
                await self.producer.publish("teaRPC", json.dumps(error))

        self.method_listeners[method] = handler

    async def introspect(self) -> list[str]:
        id_ = uuid.uuid4().hex
        methods: set[str] = set()
        fut = asyncio.get_event_loop().create_future()

        async def cb(msg: RPCMessage):
            if msg["payload"]["type"] == "introspect-response":
                for m in msg["payload"]["methods"]:
                    methods.add(m)

        self.callback_listeners[id_] = cb

        asyncio.get_event_loop().call_later(
            self.ack_timeout,
            lambda: (fut.set_result(list(methods)), self.callback_listeners.pop(id_, None)),
        )

        await self.producer.publish("teaRPC", json.dumps({
            "id": id_,
            "payload": {"type": "introspect"},
        }))

        return await fut
