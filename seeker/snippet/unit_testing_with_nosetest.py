#date: 2024-02-21T17:03:10Z
#url: https://api.github.com/gists/59a69544f14512881abab773f22ba0cf
#owner: https://api.github.com/users/mvandermeulen

#!/usr/bin/env python
import asyncio
import json
import time
import uuid
from decimal import Decimal
from io import StringIO
from os.path import join, realpath

from nose.plugins.attrib import attr
import logging
import unittest

@attr('stable')
class ExecutionReportUnitTest(unittest.TestCase):

    def setUp(self):
        self.tredis = db.get_client()
        self.redis = self.tredis.redis
        self.redis.flushdb()

    def get_server_name(self):
        return "unit-test"

    def create_sample_data(self, order_id, full=False, clock=0.0) -> dict:
        server_name = self.get_server_name()
        exchange_1 = "Test1"
        exchange_2 = "Test2"
        asset_1 = "BTC"
        asset_2 = "GBP"
        session_started_at = time.time()
        session_name = create_trade_session_name(exchange_1, exchange_2, asset_1, asset_2, session_started_at)
        order_data = {
            "order_id": order_id,
            "version": "0.0",
            "session_name": session_name,
            "server_name": server_name,
            "session_started_at": session_started_at,
            "started_at": time.time(),
            "clock": clock,
            "kind": "buy_2_sell_1",
            "quantity_1": Decimal(1),
            "quantity_2": Decimal(1),
            "market_1_bid_price": Decimal(1),
            "market_2_bid_price": Decimal(1),
            "market_1_ask_price": Decimal(1),
            "market_2_ask_price": Decimal(1),
            "expected_profitability": 1.0,
            "unit_test": True,
            "profitability_buy_1_sell_2": 1.0,
            "profitability_buy_2_sell_1": 1.0,
            "executed_quantity_1": Decimal(1),
            "executed_price_1": Decimal(1),
            "exchange_1_id": "aaa",
            "asset_1": asset_1,
            "asset_2": asset_2,
            "exchange_1": exchange_1,
            "exchange_2": exchange_2,
        }

        if full:
            order_data.update({
                "exchange_2_id": "bbb",
                "closed_at": time.time(),
                "succeed_at": time.time(),
                "order_1_latency": 100.0,
                "order_2_latency": 100.0,
                "executed_price_2": Decimal(1),
                "executed_quantity_2": Decimal(1),
                "dust_gathered": Decimal(0),
                "exchange_1_fees_paid": Decimal("1.00"),
                "exchange_2_fees_paid": Decimal("2.00"),
                "realised_profitability": 0.03,
                "realised_profitability_with_fees": 0.0175,
                "exchange_2_accepted_slippage": 3,
                "exchange_2_fok_attempts": 3,
            })

        return order_data

    def test_export_import_roundtrip(self):
        """Serialise and deserialise JSON data"""
        data = self.create_sample_data(str(uuid.uuid4()))
        report = ArbTradeExecution(**data)
        report.validate()
        json_export = report.to_json()
        json.loads(json_export)  # JSON parses correctly
        report_deserialised = ArbTradeExecution.from_json(json_export)
        # Test a single member
        assert report_deserialised.market_1_bid_price == report.market_1_bid_price


    def test_import_bad_member(self):
        """Import bad member in JSON"""
        data = self.create_sample_data(str(uuid.uuid4()))
        report = ArbTradeExecution(**data)
        json_export = report.to_json()
        bad_data = json.loads(json_export)
        bad_data["foo"] = "bar"
        bad_json = json.dumps(bad_data)
        with self.assertRaises(ValueError):
            ArbTradeExecution.from_json(bad_json)

    def test_store_bad_decimal(self):
        """Storing float is not allowed when Decimal is required"""
        data = self.create_sample_data(str(uuid.uuid4()))
        data["market_1_bid_price"] = 1.0
        report = ArbTradeExecution(**data)
        with self.assertRaises(ValueError):
            report.validate()

    def test_report_partial_trade(self):
        """See we get a stream of incoming web socket messages."""

        order_id = str(uuid.uuid4())
        order_data = self.create_sample_data(order_id)
        report = ArbTradeExecution(**order_data)
        report.validate()
        server_name = order_data["server_name"]
        order_id = order_data["order_id"]

        # Do a round trip in database
        reporter = ExecutionReportManager(self.redis)
        persistent_id = reporter.create_execution_report(report)
        execution_report_deserialised = reporter.load_execution_report(server_name, persistent_id)
        assert execution_report_deserialised.order_id == order_id

    def test_report_full_trade(self):
        """See we get a stream of incoming web socket messages."""

        order_id = str(uuid.uuid4())
        order_data = self.create_sample_data(order_id, full=True)
        report = ArbTradeExecution(**order_data)
        report.validate()
        server_name = order_data["server_name"]
        order_id = order_data["order_id"]

        # Do a round trip in database
        reporter = ExecutionReportManager(self.redis)
        persistent_id = reporter.create_execution_report(report)
        execution_report_deserialised = reporter.load_execution_report(server_name, persistent_id)
        assert execution_report_deserialised.exchange_2_accepted_slippage == 3

    def test_load_by_time(self):
        """See our time based indexing works."""
        server_name = self.get_server_name()
        trades = [
            ArbTradeExecution(**self.create_sample_data('x', clock=1)),
            ArbTradeExecution(**self.create_sample_data('y', clock=2)),
        ]
        reporter = ExecutionReportManager(self.redis)
        for t in trades:
            reporter.create_execution_report(t)
        # Load in ascending order
        trades = reporter.load_execution_reports_by_time(server_name, 0, 9999, desc=False)
        trades = list(trades)
        assert len(trades) == 2
        assert trades[0].clock == 1
        assert trades[1].clock == 2
        # Load in descending order
        trades = reporter.load_execution_reports_by_time(server_name, 0, 9999, desc=True)
        trades = list(trades)
        assert len(trades) == 2
        assert trades[0].clock == 2
        assert trades[1].clock == 1

    def test_last_trade(self):
        """We can load the last trade ordered by clock."""
        server_name = self.get_server_name()
        reporter = ExecutionReportManager(self.redis)
        last_trade = reporter.get_last_trade(server_name)
        assert last_trade is None
        trades = [
            ArbTradeExecution(**self.create_sample_data('x', clock=1)),
            ArbTradeExecution(**self.create_sample_data('z', clock=3)),
            ArbTradeExecution(**self.create_sample_data('y', clock=2)),
        ]
        for t in trades:
            reporter.create_execution_report(t)
        last_trade = reporter.get_last_trade(server_name)
        assert last_trade.order_id == "z"

    def test_export_csv(self):
        """Exporting full and partial orders to CSV works."""
        trade_list = [
            ArbTradeExecution(**self.create_sample_data(str(uuid.uuid4()), full=False)),
            ArbTradeExecution(**self.create_sample_data(str(uuid.uuid4()), full=True)),
        ]
        stream = StringIO()
        export_csv(stream, trade_list)
        data = stream.getvalue()
        assert trade_list[0].order_id in data
        assert trade_list[1].order_id in data

    def test_generate_export(self):
        """Check we generate export data correctly.."""
        reporter = ExecutionReportManager(self.redis)
        trades = [
            ArbTradeExecution(**self.create_sample_data('x', clock=1)),
            ArbTradeExecution(**self.create_sample_data('z', clock=3)),
            ArbTradeExecution(**self.create_sample_data('y', clock=2)),
        ]
        for t in trades:
            reporter.create_execution_report(t)
        stream = StringIO()
        fname, caption = generate_export(reporter, stream)
        assert fname
        assert caption
        stream.seek(0)
        assert len(stream.read()) > 500
