#date: 2024-02-21T17:03:10Z
#url: https://api.github.com/gists/59a69544f14512881abab773f22ba0cf
#owner: https://api.github.com/users/mvandermeulen

"""A simple trade execution report database by using Python dataclass, Redis and JSON.

- A very simple database for arbitrage trade execution reports.
- Dataclasses are serialised and deserialised to JSON that is stored in the Redis.
- There is type validation for members and class-types.
- Class-types like Decimal are converted back to their original format upon deserialisation.
- Optional members are supported and the member presence is validated.
- Past trades can be iterated in either order by creation.
- A simple CSV exported is provided.

Ps. I checked couple of existing dataclass validator packages on PyPi and did not find anything simple and suited
for the purpose, especially when nested items are not needed. (They cannot be needed, data goes to a spreadsheet
in the end).
"""

import datetime
import json
import socket
import time
import typing
from decimal import Decimal
from io import TextIOBase
from typing import Optional, List
from dataclasses import dataclass, asdict

from redis import StrictRedis
from dataclass_csv import DataclassWriter


@dataclass
class ArbTradeExecution:
    """Define arbitrage trade related data for later analysis."""

    # Internal ticker symbols for the pair we are trading
    asset_1: str # Always crypto
    asset_2: str # Always fiat

    # Exchange names this trade was performed on
    exchange_1: str
    exchange_2: str

    version: str  # Strategy version that procuced this trade
    order_id: str  # Trade id
    session_name: str  # Trading session name
    server_name: str # Which server runs the strategy

    # Are we in testing mode
    unit_test: bool

    #
    # Trade ordering related
    #

    session_started_at: float  # How long the bot has been running
    started_at: float  # Wall clock when the order was recorded
    clock: float  # Framework clock signal when the order was started

    # Which way was this trade, e.g. buy_2_sell_1
    kind: str

    # How much was the original target execution quantity in asset_1
    quantity_1: Decimal
    quantity_2: Decimal

    # Market state when the decision of a trade was made
    market_1_bid_price: Decimal
    market_2_bid_price: Decimal
    market_1_ask_price: Decimal
    market_2_ask_price: Decimal
    profitability_buy_1_sell_2: float
    profitability_buy_2_sell_1: float
    expected_profitability: float

    # Exchange 1 execution information.
    # Note this is always present, as if there is no exchange 1 trade there is no
    # trade at all in our data model.
    exchange_1_id: str  # Exchange 1 internal id for the order
    executed_quantity_1: Decimal
    executed_price_1: Decimal

    # Was this trade closed successful. Timestamps of closing.
    closed_at: Optional[float] = None
    succeed_at: Optional[float] = None  # Set if both trades are done
    failed_at: Optional[float] = None  # Set if only the first trade is done

    # How fast we were in seconds
    order_1_latency: Optional[float] = None
    order_2_latency: Optional[float] = None

    # Raw exchange execution response results
    order_1_data: Optional[str] = None
    order_2_data: Optional[str] = None

    # How good counter fill we got on the counter exchange
    executed_price_2: Optional[Decimal] = None
    executed_quantity_2: Optional[Decimal] = None
    exchange_2_id: Optional[str] = None  # Exchange 2 internal id for the order

    # Partial execution dust handling
    # For failed trades
    dust_gathered: Optional[Decimal] = None # For poorly executioned trades
    dust_cleared: Optional[Decimal] = None # For completed trades

    # Fees
    exchange_1_fees_paid: Optional[Decimal] = None
    exchange_2_fees_paid: Optional[Decimal] = None

    # How good was our performance
    realised_arbitration: Optional[Decimal] = None  # In fiat
    realised_profitability: Optional[float] = None
    realised_profitability_with_fees: Optional[float] = None

    # FOK order management
    exchange_2_accepted_slippage: Optional[int] = None  # In BPS
    exchange_2_fok_attempts: Optional[int] = None  # How many round of attempts we did to get a fill

    # Asset balances at the end of the trade.
    # This allows following the balance development and easily
    # pop up the last trade to read exchange balances.
    exchange_1_balance_1: Optional[Decimal] = None
    exchange_1_balance_2: Optional[Decimal] = None
    exchange_2_balance_1: Optional[Decimal] = None
    exchange_2_balance_2: Optional[Decimal] = None

    # Internal json encoder that handles the Decimal instances
    # https://stackoverflow.com/a/3885198/315168
    class _Encoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, Decimal):
                return str(o)
            return super(ArbTradeExecution._Encoder, self).default(o)

    @property
    def crypto(self) -> str:
        """Get crypto asset name of this trading paid."""
        return self.asset_1

    @property
    def fiat(self) -> str:
        """Get fiat asset name of this trading paid."""
        return self.asset_2

    @property
    def quantity_crypto(self) -> Decimal:
        return self.quantity_1

    @property
    def quantity_fiat(self) -> Decimal:
        return self.quantity_2

    def validate(self):
        """Validate instance contents against the Python type hints."""
        for name in self.__dataclass_fields__:

            optional, expected_type = self.get_member_info(name)

            value = self.__dict__.get(name)
            if value is None and optional:
                # Optional member was not filled in
                continue

            actual_type = type(value)

            if expected_type != actual_type:
                raise ValueError(f"Field {name} expected type {expected_type}, got {actual_type}")

    def to_json(self) -> str:
        return json.dumps(asdict(self), cls=ArbTradeExecution._Encoder)

    @classmethod
    def get_member_info(cls, name) -> typing.Tuple[bool, Optional[type]]:
        """Introspect what type we expect for a dataclass member.

        :return: (optional, type)
        """

        member = cls.__dataclass_fields__.get(name)
        if member is None:
            # Unknown member
            return False, None

        expected_type = member.type

        # https://stackoverflow.com/a/66117226/315168
        optional = False
        expected_type_internal = expected_type

        if typing.get_origin(expected_type) == typing.Union:
            # Handle Optional typing
            # Internally Python expresses Optional[float] as Typing.Union[float, NoneType]
            args = typing.get_args(expected_type)
            if len(args) == 2 and args[1] == type(None):
                optional = True
                expected_type_internal = args[0]

        return optional, expected_type_internal

    @classmethod
    def from_json(cls, text: str) -> "ArbTradeExecution":
        """Load JSON blob back to Python data and convert all memberes to objects."""
        data = json.loads(text)
        prepared_data = {}
        for name in data:
            if name in data:
                # Convert back to decimals
                optional, member_type = cls.get_member_info(name)
                if member_type is None:
                    raise ValueError(f"JSON had an unknown member: {name}")

                value = data[name]
                if optional and value is None:
                    prepared_data[name] = None
                else:
                    prepared_data[name] = member_type(value)

        report = ArbTradeExecution(**prepared_data)
        report.validate()
        return report


class ExecutionReportManager:
    """Persistently store trade data in Redis.

    Uses Redis storage, where data is partitioned by server name and order id using HSETs.

    We use hash sets (HSET), one per server. This allows us move and manipulate per-server
    data more easily.
    """

    #: Hash map for the reports
    HKEY_PREFIX = "execution_report"

    #: Time index for the reports
    ZKEY_PREFIX = "execution_report_sorted"

    def __init__(self, redis: StrictRedis):
        self.redis = redis

    def create_execution_report(self, report: ArbTradeExecution) -> str:
        """Store an execution report in the database.

        Simple append only data structure - overwrites especially unwanted.
        """
        server_name = report.server_name
        assert server_name
        hkey = f"{self.HKEY_PREFIX}:{server_name}"
        zkey = f"{self.ZKEY_PREFIX}:{server_name}"
        key = report.order_id
        data = report.to_json()
        if self.redis.hexists(hkey, key):
            # Check the existince of the report
            # to make problematic code bark out loud
            raise RuntimeError(f"Execution report {key} has already been written to the databased")
        # Store the actual report content
        # TODO: We really do not care about atomicity guarantees here,
        # as we assume only one writer
        self.redis.hset(hkey, key, data)
        # Manage sorted index of keys
        # https://redis.io/commands/ZADD
        self.redis.zadd(zkey, {key: report.clock})
        return key

    def load_execution_report(self, server_name: str, look_up_key: str) -> ArbTradeExecution:
        hkey = f"{self.HKEY_PREFIX}:{server_name}"
        text = self.redis.hget(hkey, look_up_key)
        return ArbTradeExecution.from_json(text)

    def load_execution_reports_by_time(self, server_name, start, end, desc=False) -> typing.Iterable[ArbTradeExecution]:
        """Allow iterating through all execution reports in the database in order they were created.

        You can itereate either oldest first or newest first.
        The trades are sorted by clock.
        """
        zkey = f"{self.ZKEY_PREFIX}:{server_name}"
        # Load a span of reports based by the time index
        keys = self.redis.zrange(zkey, start, end, desc=desc)
        for key in keys:
            # Load individual reports
            yield self.load_execution_report(server_name, key)

    def get_first_trade(self, server_name: str) -> typing.Optional[ArbTradeExecution]:
        """Convenience method to the peek the first executed trade."""
        for report in self.load_execution_reports_by_time(server_name, start=0, end=1, desc=False):
            return report
        return None

    def get_last_trade(self, server_name: str) -> typing.Optional[ArbTradeExecution]:
        """Convenience method to the peek the last executed trade."""
        for report in self.load_execution_reports_by_time(server_name, start=0, end=1, desc=True):
            return report
        return None

    def export_all_data(self) -> List[ArbTradeExecution]:
        """Export all servers and all execution reports."""
        entries = []
        # All servers
        for hkey in self.redis.keys(pattern=f"{self.HKEY_PREFIX}:*"):
            # All trades for a server
            for order_id in self.redis.hgetall(hkey):
                text = self.redis.hget(hkey, order_id)
                entries.append(ArbTradeExecution.from_json(text))
        return entries

    @staticmethod
    def get_server_name():
        return socket.gethostname()


def create_trade_session_name(exchange_1: str, exchange_2: str, asset_1: str, asset_2: str, started_at: float):
    """Add human readable name for trading sessions."""
    human_date = datetime.datetime.fromtimestamp(int(started_at)).strftime('%Y-%m-%d %H:%M:%S')
    return f"Arb session {exchange_1}-{exchange_2} {asset_1}-{asset_2} started at {human_date}"


def export_csv(stream: TextIOBase, trades: List[ArbTradeExecution]):
    """Create a CSV export of all trades."""
    # Sort trades by start time
    trades = sorted(trades, key=lambda t: t.clock)
    # https://pypi.org/project/dataclass-csv/
    w = DataclassWriter(stream, trades, ArbTradeExecution)
    w.write()


def generate_export(manager: ExecutionReportManager, stream: TextIOBase) -> typing.Tuple[Optional[str], Optional[str]]:
    """Export CSV data to given Python file-like stream.

    Useful e.g. to generate Telegram document upload.

    :return: Tuple (fname, caption)
    """

    # Generate in-memory CSV
    trades = manager.export_all_data()
    if not trades:
        # Nothing to export
        return None, None

    export_csv(stream, trades)

    first_date = datetime.datetime.fromtimestamp(int(trades[0].clock)).strftime('%Y-%m-%d')
    last_date = datetime.datetime.fromtimestamp(int(trades[-1].clock)).strftime('%Y-%m-%d')
    caption = f"Export of {len(trades)} trades in {first_date} - {last_date}"

    # Send file to Telegram chat via multipart/form-data
    human_date = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d')
    fname = f"trade-export-{human_date}.csv"
    return fname, caption

