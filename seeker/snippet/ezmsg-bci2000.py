#date: 2022-12-14T16:54:35Z
#url: https://api.github.com/gists/2ca805d1037aea4f575a912771ecdf35
#owner: https://api.github.com/users/griffinmilsap

import json
import uuid
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

import ezmsg.core as ez
from ezmsg.websocket import WebsocketClient, WebsocketSettings
from ezmsg.sigproc.messages import TSMessage

from typing import Callable, Dict, List, Optional, AsyncGenerator, Any


class BCI2000Message:
    """ A superclass for all BCI2000 Messages for typing purposes"""
    ...


@dataclass
class PhysicalUnit:
    offset: float
    gain: float
    symbol: str
    vmin: float
    vmax: float


@dataclass
class SignalProperties(BCI2000Message):
    name: str
    channels: List[str]
    elements: List[str]
    signal_type: str
    channel_dim_unit: PhysicalUnit
    channel_dim_trivial: bool
    element_dim_unit: PhysicalUnit
    element_dim_trivial: bool
    channel_unit: Dict[str, PhysicalUnit]


@dataclass
class StateInfo:
    width: int
    default: int
    byte: int
    bit: int


@dataclass
class GenericSignal(BCI2000Message):
    data: np.ndarray


@dataclass
class StateFormat(BCI2000Message):
    info: Dict[str, StateInfo]


@dataclass
class StateVector(BCI2000Message):
    bits: np.ndarray


def _nts(buf: bytes, start_idx: int = 0):
    """ Parse a null-terminated string and return index just after null """
    loc = start_idx + buf[start_idx:].find(0x00)
    return buf[start_idx: loc].decode('utf-8'), loc + 1


def decode(msg: bytes) -> BCI2000Message:
    return decode_fn(msg)(msg)


def decode_fn(msg: bytes) -> Callable:
    """ Reads the message descriptor and supplement,
    then returns the appropriate decode function """

    descriptor = msg[0]

    if descriptor == 3:
        return decode_StateFormat
    elif descriptor == 4:
        supplement = msg[1]
        if supplement == 1:
            return decode_GenericSignal
        elif supplement == 3:
            return decode_SignalProperties
        else:
            raise Exception("Unknown supplement")
    elif descriptor == 5:
        return decode_StateVector

    else:
        raise Exception("Unknown descriptor")


def decode_SignalProperties(msg: bytes) -> SignalProperties:

    # Sanity check
    assert decode_fn(msg) == decode_SignalProperties

    msg = msg[2:]

    # There may be better ways to do this
    # But this is adapted from bci2k.js
    # where we had to fix a bunch of bugs with
    # previous versions of BCI2000
    prop_str = msg.decode('utf-8').strip()

    # There's not always a space after a bracket
    # So let's fix that
    prop_str = prop_str.replace('{', ' { ')
    prop_str = prop_str.replace('}', ' } ')

    # Split with no args gets rid of multiple spaces
    token = "**********"

    name = "**********"

    def parse_dim():
        trivial = False
        dim_labels = []
        dim_len = "**********"
        if dim_len == '{':
            while True:
                dim_labels.append(next(token))
                if dim_labels[-1] == '}':
                    dim_labels.pop()
                    break
            dim_len = len(dim_labels)
        else:
            trivial = True
            dim_len = int(dim_len)
            dim_labels = [str(i + 1) for i in range(dim_len)]

        return dim_labels, trivial

    def parse_physical_unit():
        offset = "**********"
        gain = "**********"
        symbol = "**********"
        vmin = "**********"
        vmax = "**********"
        return PhysicalUnit(
            offset=offset,
            gain=gain,
            symbol=symbol,
            vmin=vmin,
            vmax=vmax
        )

    channels, channel_trivial = parse_dim()
    elements, element_trivial = parse_dim()

    signal_type = "**********"

    channel_dim_unit = parse_physical_unit()
    element_dim_unit = parse_physical_unit()

    next(token)  # }

    channel_unit = {}
    for ch_name in channels:
        channel_unit[ch_name] = parse_physical_unit()

    return SignalProperties(
        name=name,
        channels=channels,
        elements=elements,
        signal_type=signal_type,
        channel_dim_unit=channel_dim_unit,
        channel_dim_trivial=channel_trivial,
        element_dim_unit=element_dim_unit,
        element_dim_trivial=element_trivial,
        channel_unit=channel_unit
    )


def decode_GenericSignal(msg: bytes) -> GenericSignal:

    # Sanity check
    assert decode_fn(msg) == decode_GenericSignal
    msg = msg[2:]

    signal_type = np.int16
    if msg[0] == 1:
        raise Exception("This packet uses an unsupported Float24 datatype")
    elif msg[0] == 2:
        signal_type = np.float32
    elif msg[0] == 3:
        signal_type = np.int32

    idx = 3
    num_channels = int.from_bytes(msg[idx - 2:idx], byteorder='little', signed=False)
    if num_channels != 0xFFFF:
        idx += 2
    else:
        ch_str, idx = _nts(msg, idx)
        num_channels = int(ch_str)

    num_elements = int.from_bytes(msg[idx - 2:idx], byteorder='little', signed=False)
    if num_elements != 0xFFFF:
        idx += 2
    else:
        el_str, idx = _nts(msg, idx)
        num_elements = int(el_str)

    buf: npt.NDArray = np.frombuffer(msg[5:], dtype=signal_type)
    return GenericSignal(data=buf.reshape(num_channels, num_elements))


def decode_StateFormat(msg: bytes) -> StateFormat:

    # Sanity check
    assert decode_fn(msg) == decode_StateFormat
    msg = msg[1:]

    state_format = {}
    format_str = msg.decode('utf-8').strip()

    for state_str in format_str.split('\n'):
        name, width, default, byte, bit = tuple(state_str.split(' '))
        state_format[name] = StateInfo(
            width=int(width),
            default=int(default),
            byte=int(byte),
            bit=int(bit)
        )

    return StateFormat(info=state_format)


def decode_StateVector(msg: bytes) -> StateVector:

    # Sanity check
    assert decode_fn(msg) == decode_StateVector
    msg = msg[1:]

    vec_length, idx = _nts(msg)
    num_vecs, idx = _nts(msg, idx)

    arr: npt.NDArray = np.frombuffer(msg[idx:], dtype=np.uint8)
    arr = arr.reshape(int(num_vecs), int(vec_length))

    return StateVector(
        bits=np.unpackbits(
            arr,
            axis=1,
            bitorder='little'
        )
    )


class BCI2000InterpreterState(ez.State):
    signal_properties: Optional[SignalProperties] = None
    state_format: Optional[StateFormat] = None
    bit_slices: Dict[str, slice] = field(default_factory=dict)
    state_names: List[str] = field(default_factory=list)

    signal_info_dirty: bool = True
    state_info_dirty: bool = True


_mult = 2 ** np.arange(32)
def _bits_to_state(bits): return (bits * _mult[:bits.shape[1]]).sum(axis=1)


class BCI2000Interpreter(ez.Unit):
    """
    Interpret BCI2000 WSIOFilters and generate EEGMessage streams
    TODO: It's possible the element dim unit isn't time in seconds...
    IF SO, we need to flatten the data and publish messages with
    time axis of length 1 unit.  In this case, the sampling rate
    would be equal to the SampleBlockSize / SamplingRate
    This would only happen if we were receiving channel spectra in each message
    as might happen if we were to subscribe to the features datastream
    For now though, let's assume the element dim is our time axis, as is the case
    with the raw data coming out of the WSSourceFilter.
    As it stands, this module is only compatible with the WSSourceFilter
    """

    STATE: BCI2000InterpreterState

    INPUT = ez.InputStream(bytes)

    # FIXME: State isn't really EEG...  Maybe this needs to be renamed?
    OUTPUT_STATE = ez.OutputStream(TSMessage)
    OUTPUT_SIGNAL = ez.OutputStream(TSMessage)

    @ez.subscriber(INPUT)
    @ez.publisher(OUTPUT_STATE)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_message(self, message: bytes) -> AsyncGenerator:

        decoded = decode(message)
        if isinstance(decoded, SignalProperties):
            self.STATE.signal_properties = decoded
            self.STATE.signal_info_dirty = True

        elif isinstance(decoded, StateFormat):
            """ StateFormat always follows SignalProperties """
            self.STATE.state_format = decoded

            self.STATE.bit_slices.clear()
            for state, fmt in self.STATE.state_format.info.items():
                offset = fmt.byte * 8 + fmt.bit
                self.STATE.bit_slices[state] = slice(offset, offset + fmt.width)

            self.STATE.state_names = [v for v in sorted(self.STATE.state_format.info.keys()) if v != '__pad0']

        elif isinstance(decoded, GenericSignal):
            if self.STATE.signal_properties is None:
                return

            fs = 1.0 / self.STATE.signal_properties.element_dim_unit.gain
            ch_names = self.STATE.signal_properties.channels  \
                if not self.STATE.signal_properties.channel_dim_trivial \
                else None

            # FIXME: This outputs FORTRAN order arrays
            out_msg = TSMessage(decoded.data.T, fs=fs)
            # Backward compat with old EEGMessage
            setattr(out_msg, 'ch_names', ch_names)
            setattr(out_msg, 'ch_dim', 1)
            yield (self.OUTPUT_SIGNAL, out_msg)

        elif isinstance(decoded, StateVector):
            if self.STATE.signal_properties is None:
                return
            if self.STATE.state_format is None:
                return

            fs = 1.0 / self.STATE.signal_properties.element_dim_unit.gain
            ch_names = self.STATE.state_names

            state_arr = []
            for state in self.STATE.state_names:
                sl = self.STATE.bit_slices[state]
                state_bits = decoded.bits[:, sl]
                state_arr.append(_bits_to_state(state_bits[:-1, :]))

            # FIXME: This outputs FORTRAN order arrays
            out_msg = TSMessage(np.array(state_arr).T, fs=fs)
            # Backward compat with old EEGMessage
            setattr(out_msg, 'ch_names', ch_names)
            setattr(out_msg, 'ch_dim', 1)

            yield (self.OUTPUT_STATE, out_msg)


class BCI2000WSIO(ez.Collection):

    SETTINGS: WebsocketSettings

    OUTPUT_STATE = ez.OutputStream(TSMessage)
    OUTPUT_SIGNAL = ez.OutputStream(TSMessage)

    CLIENT = WebsocketClient()
    INTERPRETER = BCI2000Interpreter()

    def configure(self) -> None:
        self.CLIENT.apply_settings(self.SETTINGS)

    def network(self) -> ez.NetworkDefinition:
        return [
            (self.CLIENT.OUTPUT, self.INTERPRETER.INPUT),
            (self.INTERPRETER.OUTPUT_STATE, self.OUTPUT_STATE),
            (self.INTERPRETER.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL)
        ]

@dataclass
class BCI2000WebMessage:
    command: str
    output: Optional[str] = None
    id: uuid.UUID = field(default_factory=uuid.uuid1)


class BCI2000WebState(ez.State):
    id_counter: int = 0
    message_dict: Dict[int, BCI2000WebMessage] = field(default_factory=dict)


class BCI2000WebInterpreter(ez.Unit):

    STATE: BCI2000WebState

    COMMAND = ez.InputStream(BCI2000WebMessage)
    RESULT = ez.OutputStream(BCI2000WebMessage)

    WSTX = ez.OutputStream(bytes)
    WSRX = ez.InputStream(bytes)

    @ez.subscriber(COMMAND)
    @ez.publisher(WSTX)
    async def on_command(self, message: BCI2000WebMessage) -> AsyncGenerator:
        msg_id = self.STATE.id_counter

        self.STATE.id_counter += 1
        self.STATE.message_dict[msg_id] = message

        yield self.WSTX, json.dumps(dict(
            opcode='E',
            id=msg_id,
            contents=message.command
        )).encode('utf-8')

    @ez.subscriber(WSRX)
    @ez.publisher(RESULT)
    async def on_output(self, message: bytes) -> AsyncGenerator:
        obj = json.loads(message)
        try:
            opcode = obj['opcode']
            id = obj['id']
            bcimsg = self.STATE.message_dict[id]
            if opcode == 'O':
                bcimsg.output = obj['contents']

            yield self.RESULT, bcimsg
        except BaseException:
            raise Exception('Error interpreting BCI2K message')


class BCI2000WebLogSettings(ez.Settings):
    name: str = 'ezbci2000web'
    formatter: Callable[[Any], str] = repr


class BCI2000WebLog(ez.Unit):

    SETTINGS: BCI2000WebLogSettings

    INPUT = ez.InputStream(Any)
    COMMAND = ez.OutputStream(BCI2000WebMessage)

    @ez.subscriber(INPUT)
    @ez.publisher(COMMAND)
    async def echo(self, message: Any) -> AsyncGenerator:
        out = self.SETTINGS.formatter(message)
        cmd = f'LOG {self.SETTINGS.name}: {out}'

        yield self.COMMAND, BCI2000WebMessage(
            command=cmd
        )


class BCI2000Web(ez.Collection):

    SETTINGS: WebsocketSettings

    # Execute an arbitrary command
    COMMAND = ez.InputStream(BCI2000WebMessage)

    # Print to BCI2000 Operator Log
    LOG = ez.InputStream(Any)

    # Output from any command sent to BCI2K
    RESULT = ez.OutputStream(BCI2000WebMessage)

    WSCLIENT = WebsocketClient()
    INTERPRETER = BCI2000WebInterpreter()
    LOGGER = BCI2000WebLog()

    def configure(self) -> None:
        self.WSCLIENT.apply_settings(self.SETTINGS)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.COMMAND, self.INTERPRETER.COMMAND),
            (self.LOG, self.LOGGER.INPUT),
            (self.LOGGER.COMMAND, self.INTERPRETER.COMMAND),
            (self.INTERPRETER.WSTX, self.WSCLIENT.INPUT),
            (self.WSCLIENT.OUTPUT, self.INTERPRETER.WSRX),
            (self.INTERPRETER.RESULT, self.RESULT)
        )

# Dev / Test

from ezmsg.testing.debuglog import DebugLog

class BCI2000Log(ez.Collection):

    SETTINGS: WebsocketSettings

    BCI2000 = BCI2000WSIO()
    DEBUG = DebugLog()

    def configure(self) -> None:
        self.BCI2000.apply_settings(self.SETTINGS)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.BCI2000.OUTPUT_SIGNAL, self.DEBUG.INPUT),
        )


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser('Record raw BCI2000 data')

    parser.add_argument(
        '--bci2000host',
        type=str,
        help='Hostname for BCI2000 WSIOSourceFilter',
        default='localhost'
    )

    parser.add_argument(
        '--wsioport',
        type=int,
        help='Port for BCI2000 WSIOSourceFilter',
        default=20100
    )

    class Args:
        bci2000host: str
        wsioport: int

    args = parser.parse_args(namespace=Args)

    wsio_host: str = args.bci2000host
    wsio_port: int = args.wsioport

    settings = WebsocketSettings(host=wsio_host, port=wsio_port)
    system = BCI2000Log(settings)

    ez.run(system) BCI2000Log(settings)

    ez.run(system)