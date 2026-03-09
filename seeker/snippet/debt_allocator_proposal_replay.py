#date: 2026-03-09T17:22:52Z
#url: https://api.github.com/gists/f43a708046828ccd8273ac3d13a9099f
#owner: https://api.github.com/users/adpthegreat

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from brownie import chain, network, web3
from hexbytes import HexBytes
from web3 import Web3
from web3._utils.events import get_event_data

try:
    from scripts.constants import MAX_BPS
    from scripts.utils.contract import get_contract
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts.constants import MAX_BPS
    from scripts.utils.contract import get_contract

DEFAULT_ALLOCATOR = "0x1e9eB053228B1156831759401dE0E115356b8671"
DEFAULT_APPLICATOR = "0x6b1FC0c4370ee907220c3Af561940d3E21081f9B"
DEFAULT_APR_ORACLE = "0x1981AD9F44F2EA9aDd2dC4AD7D075c102C70aF92"
DEFAULT_VAULT = "0x182863131F9a4630fF9E27830d945B1413e347E8"
DEFAULT_MULTICALL3 = "0xcA11bde05977b3631167028862bE2a173976CA11"
DEFAULT_LOG_CHUNK = 75_000
DEFAULT_MULTICALL_BATCH = 120

NAME_ABI = [
    {
        "inputs": [],
        "name": "name",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    }
]


@dataclass
class CallSpec:
    key: Any
    contract: Any
    fn_name: str
    args: Tuple[Any, ...]
    allow_failure: bool = True


def _chunked(values: List[Any], size: int) -> Iterable[List[Any]]:
    for idx in range(0, len(values), size):
        yield values[idx : idx + size]


def _addr_topic(address: str) -> str:
    cleaned = address.lower().replace("0x", "")
    return "0x" + ("0" * 24) + cleaned


def _short(address: str) -> str:
    return f"{address[:6]}...{address[-4:]}"


def _pct_from_1e18(apr_1e18: int) -> Decimal:
    return (Decimal(apr_1e18) * Decimal(100)) / Decimal(10**18)


def _fmt_pct(value: Decimal) -> str:
    return f"{value:.2f}%"


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    return int(value)


def _decode_output(output_types: List[str], raw: bytes) -> Any:
    decoded = web3.codec.decode(output_types, raw)
    if len(decoded) == 1:
        return decoded[0]
    return decoded


def _canonical_abi_type(item: Dict[str, Any]) -> str:
    abi_type = item["type"]
    if not abi_type.startswith("tuple"):
        return abi_type

    components = item.get("components", [])
    inner = ",".join(_canonical_abi_type(component) for component in components)
    suffix = abi_type[len("tuple") :]
    return f"({inner}){suffix}"


def _direct_read(specs: List[CallSpec], block_number: int) -> Dict[Any, Any]:
    results: Dict[Any, Any] = {}
    for spec in specs:
        try:
            fn = getattr(spec.contract.functions, spec.fn_name)(*spec.args)
            results[spec.key] = fn.call(block_identifier=block_number)
        except Exception:
            results[spec.key] = None
    return results


def multicall_read(
    multicall: Any,
    specs: List[CallSpec],
    block_number: int,
    batch_size: int,
) -> Tuple[Dict[Any, Any], bool]:
    if not specs:
        return {}, False

    results: Dict[Any, Any] = {}
    try:
        for batch in _chunked(specs, batch_size):
            calls = []
            decode_meta: List[Tuple[Any, List[str]]] = []

            for spec in batch:
                fn = getattr(spec.contract.functions, spec.fn_name)(*spec.args)
                call_data = fn._encode_transaction_data()
                output_types = [
                    _canonical_abi_type(item) for item in fn.abi["outputs"]
                ]
                calls.append((spec.contract.address, spec.allow_failure, call_data))
                decode_meta.append((spec.key, output_types))

            raw_results = multicall.functions.aggregate3(calls).call(
                block_identifier=block_number
            )

            for idx, one in enumerate(raw_results):
                success = bool(one[0])
                return_data = one[1]
                key, output_types = decode_meta[idx]

                if not success:
                    results[key] = None
                    continue

                results[key] = _decode_output(output_types, return_data)

        return results, False
    except Exception:
        return _direct_read(specs, block_number), True


def find_contract_deployment_block(address: str) -> int:
    latest = web3.eth.block_number
    code_latest = web3.eth.get_code(address, block_identifier=latest)
    if code_latest in (b"", HexBytes("0x")):
        raise ValueError(f"no contract bytecode at latest block for {address}")

    low = 0
    high = latest
    while low < high:
        mid = (low + high) // 2
        code = web3.eth.get_code(address, block_identifier=mid)
        if code in (b"", HexBytes("0x")):
            low = mid + 1
        else:
            high = mid
    return low


def fetch_ratio_update_logs(
    allocator: Any,
    vault: str,
    from_block: int,
    to_block: int,
    log_chunk: int,
) -> Dict[str, Dict[str, Any]]:
    allocator_addr = allocator.address
    event_abi = next(
        item
        for item in allocator.abi
        if item.get("type") == "event" and item.get("name") == "UpdateStrategyDebtRatio"
    )

    event_topic = Web3.to_hex(Web3.keccak(text="UpdateStrategyDebtRatio(address,address,uint256,uint256,uint256)"))
    vault_topic = _addr_topic(vault)

    grouped: Dict[str, Dict[str, Any]] = {}

    cursor = from_block
    while cursor <= to_block:
        end = min(cursor + log_chunk - 1, to_block)
        raw_logs = web3.eth.get_logs(
            {
                "address": allocator_addr,
                "fromBlock": cursor,
                "toBlock": end,
                "topics": [event_topic, vault_topic],
            }
        )

        for raw in raw_logs:
            decoded = get_event_data(web3.codec, event_abi, raw)
            tx_hash = raw["transactionHash"].hex()

            entry = grouped.setdefault(
                tx_hash,
                {
                    "tx_hash": tx_hash,
                    "block_number": int(raw["blockNumber"]),
                    "transaction_index": int(raw["transactionIndex"]),
                    "updates": [],
                },
            )

            entry["updates"].append(
                {
                    "log_index": int(raw["logIndex"]),
                    "strategy": Web3.to_checksum_address(decoded["args"]["strategy"]),
                    "new_target_ratio": int(decoded["args"]["newTargetRatio"]),
                    "new_max_ratio": int(decoded["args"]["newMaxRatio"]),
                    "new_total_debt_ratio": int(decoded["args"]["newTotalDebtRatio"]),
                }
            )

        cursor = end + 1

    for item in grouped.values():
        item["updates"].sort(key=lambda update: update["log_index"])

    return grouped


def _maybe_name(value: Any, fallback_address: str) -> str:
    if value is None:
        return _short(fallback_address)

    if isinstance(value, bytes):
        try:
            decoded = value.decode("utf-8", errors="ignore").replace("\x00", "").strip()
            return decoded if decoded else _short(fallback_address)
        except Exception:
            return _short(fallback_address)

    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else _short(fallback_address)

    return _short(fallback_address)


def analyze_tx(
    tx_item: Dict[str, Any],
    vault: Any,
    allocator: Any,
    apr_oracle: Any,
    multicall: Any,
    vault_address: str,
    multicall_batch: int,
) -> Tuple[Dict[str, Any], bool]:
    block_number = tx_item["block_number"]

    changed_ratios: Dict[str, int] = {}
    changed_strategies: List[str] = []
    for update in tx_item["updates"]:
        strategy = Web3.to_checksum_address(update["strategy"])
        changed_ratios[strategy] = int(update["new_target_ratio"])
        changed_strategies.append(strategy)

    queue = vault.functions.get_default_queue().call(block_identifier=block_number)
    queue = [Web3.to_checksum_address(strategy) for strategy in queue]

    strategy_candidates = list(dict.fromkeys(queue + changed_strategies))

    total_assets_key = ("total_assets",)
    before_specs: List[CallSpec] = [
        CallSpec(total_assets_key, vault, "totalAssets", tuple(), allow_failure=False)
    ]

    strategy_name_contracts: Dict[str, Any] = {}
    for strategy in strategy_candidates:
        strategy_name_contracts[strategy] = web3.eth.contract(
            address=Web3.to_checksum_address(strategy), abi=NAME_ABI
        )
        before_specs.extend(
            [
                CallSpec(("state", strategy), vault, "strategies", (strategy,), True),
                CallSpec(
                    ("ratio_after", strategy),
                    allocator,
                    "getStrategyTargetRatio",
                    (vault_address, strategy),
                    True,
                ),
                CallSpec(
                    ("apr_before", strategy),
                    apr_oracle,
                    "getStrategyApr",
                    (strategy, 0),
                    True,
                ),
                CallSpec(("name", strategy), strategy_name_contracts[strategy], "name", tuple(), True),
            ]
        )

    before_results, used_fallback = multicall_read(
        multicall=multicall,
        specs=before_specs,
        block_number=block_number,
        batch_size=multicall_batch,
    )

    total_assets = _to_int(before_results.get(total_assets_key))

    selected_strategies: List[str] = []
    for strategy in strategy_candidates:
        state = before_results.get(("state", strategy))
        current_debt = _to_int(state[2]) if state is not None else 0
        if current_debt > 0 or strategy in changed_ratios:
            selected_strategies.append(strategy)

    ratio_before_block = max(block_number - 1, 0)
    ratio_before_specs: List[CallSpec] = []
    for strategy in selected_strategies:
        ratio_before_specs.append(
            CallSpec(
                ("ratio_before", strategy),
                allocator,
                "getStrategyTargetRatio",
                (vault_address, strategy),
                True,
            )
        )

    ratio_before_results, fallback_ratio_before = multicall_read(
        multicall=multicall,
        specs=ratio_before_specs,
        block_number=ratio_before_block,
        batch_size=multicall_batch,
    )
    used_fallback = used_fallback or fallback_ratio_before

    strategy_rows: List[Dict[str, Any]] = []
    apr_after_specs: List[CallSpec] = []

    for strategy in selected_strategies:
        state = before_results.get(("state", strategy))
        current_debt = _to_int(state[2]) if state is not None else 0

        ratio_after = _to_int(before_results.get(("ratio_after", strategy)))
        ratio_before = _to_int(ratio_before_results.get(("ratio_before", strategy)))

        target_debt = (
            (total_assets * ratio_after) // MAX_BPS if total_assets > 0 else 0
        )
        nominal_delta = int(target_debt) - int(current_debt)

        apr_after_specs.append(
            CallSpec(
                ("apr_after", strategy),
                apr_oracle,
                "getStrategyApr",
                (strategy, nominal_delta),
                True,
            )
        )

        strategy_rows.append(
            {
                "strategy": strategy,
                "name": _maybe_name(before_results.get(("name", strategy)), strategy),
                "current_debt": current_debt,
                "target_debt": target_debt,
                "nominal_delta": nominal_delta,
                "ratio_before_bps": ratio_before,
                "ratio_after_bps": ratio_after,
                "apr_before_1e18": _to_int(before_results.get(("apr_before", strategy))),
                "changed": strategy in changed_ratios,
            }
        )

    apr_after_results, fallback_apr_after = multicall_read(
        multicall=multicall,
        specs=apr_after_specs,
        block_number=block_number,
        batch_size=multicall_batch,
    )
    used_fallback = used_fallback or fallback_apr_after

    vault_before_apr_pct = Decimal(0)
    vault_after_apr_pct = Decimal(0)

    for row in strategy_rows:
        strategy = row["strategy"]
        apr_after_1e18 = _to_int(apr_after_results.get(("apr_after", strategy)))
        row["apr_after_1e18"] = apr_after_1e18

        before_apr_pct = _pct_from_1e18(row["apr_before_1e18"])
        after_apr_pct = _pct_from_1e18(row["apr_after_1e18"])

        before_ratio_pct = Decimal(0)
        after_ratio_pct = Decimal(0)
        if total_assets > 0:
            before_ratio_pct = (Decimal(row["current_debt"]) * Decimal(100)) / Decimal(total_assets)
            after_ratio_pct = (Decimal(row["target_debt"]) * Decimal(100)) / Decimal(total_assets)

        before_contrib_pct = (before_apr_pct * before_ratio_pct) / Decimal(100)
        after_contrib_pct = (after_apr_pct * after_ratio_pct) / Decimal(100)

        row["apr_before_pct"] = float(before_apr_pct)
        row["apr_after_pct"] = float(after_apr_pct)
        row["debt_ratio_before_pct"] = float(before_ratio_pct)
        row["debt_ratio_after_pct"] = float(after_ratio_pct)
        row["apr_contribution_before_pct"] = float(before_contrib_pct)
        row["apr_contribution_after_pct"] = float(after_contrib_pct)

        vault_before_apr_pct += before_contrib_pct
        vault_after_apr_pct += after_contrib_pct

    strategy_rows.sort(
        key=lambda row: (
            0 if row["changed"] else 1,
            -abs(row["nominal_delta"]),
            row["name"].lower(),
        )
    )

    return (
        {
            "tx_hash": tx_item["tx_hash"],
            "block_number": block_number,
            "timestamp": int(web3.eth.get_block(block_number)["timestamp"]),
            "is_batched": len(tx_item["updates"]) > 1,
            "changed_updates": tx_item["updates"],
            "changed_strategy_count": len({item["strategy"] for item in tx_item["updates"]}),
            "selected_strategy_count": len(strategy_rows),
            "total_assets": total_assets,
            "vault_apr_before_pct": float(vault_before_apr_pct),
            "vault_apr_after_pct": float(vault_after_apr_pct),
            "strategies": strategy_rows,
        },
        used_fallback,
    )


def print_proposal_summary(vault_name: str, proposal: Dict[str, Any]) -> None:
    block_ts = datetime.fromtimestamp(proposal["timestamp"], tz=timezone.utc).isoformat()
    before = Decimal(str(proposal["vault_apr_before_pct"]))
    after = Decimal(str(proposal["vault_apr_after_pct"]))
    delta = after - before

    print(
        f"\n{vault_name} proposal tx {proposal['tx_hash']}"
    )
    print(
        f"  block {proposal['block_number']} @ {block_ts} | batched={proposal['is_batched']} | strategies={proposal['selected_strategy_count']}"
    )
    print(f"  Vault APR: {_fmt_pct(before)} => {_fmt_pct(after)} ({_fmt_pct(delta)})")

    for row in proposal["strategies"]:
        before_ratio = Decimal(str(row["debt_ratio_before_pct"]))
        after_ratio = Decimal(str(row["debt_ratio_after_pct"]))
        before_apr = Decimal(str(row["apr_before_pct"]))
        after_apr = Decimal(str(row["apr_after_pct"]))
        before_contrib = Decimal(str(row["apr_contribution_before_pct"]))
        after_contrib = Decimal(str(row["apr_contribution_after_pct"]))

        changed_flag = "*" if row["changed"] else " "
        print(
            f"  {changed_flag} {row['name']} ({row['strategy']}): {_fmt_pct(before_ratio)} => {_fmt_pct(after_ratio)}"
        )
        print(
            f"      APR {_fmt_pct(before_apr)} => {_fmt_pct(after_apr)} | contribution {_fmt_pct(before_contrib)} => {_fmt_pct(after_contrib)}"
        )


def write_outputs(
    proposals: List[Dict[str, Any]],
    out_prefix: str,
    metadata: Dict[str, Any],
) -> Tuple[str, str]:
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    json_path = f"{out_prefix}.json"
    csv_path = f"{out_prefix}.csv"

    with open(json_path, "w", encoding="utf-8") as handle:
        payload = {
            "meta": metadata,
            "proposals": proposals,
        }
        json.dump(payload, handle, indent=2)

    csv_fields = [
        "tx_hash",
        "block_number",
        "timestamp",
        "is_batched",
        "vault_total_assets",
        "vault_apr_before_pct",
        "vault_apr_after_pct",
        "strategy",
        "name",
        "changed",
        "ratio_before_bps",
        "ratio_after_bps",
        "debt_ratio_before_pct",
        "debt_ratio_after_pct",
        "current_debt",
        "target_debt",
        "nominal_delta",
        "apr_before_1e18",
        "apr_after_1e18",
        "apr_before_pct",
        "apr_after_pct",
        "apr_contribution_before_pct",
        "apr_contribution_after_pct",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()

        for proposal in proposals:
            for row in proposal["strategies"]:
                writer.writerow(
                    {
                        "tx_hash": proposal["tx_hash"],
                        "block_number": proposal["block_number"],
                        "timestamp": proposal["timestamp"],
                        "is_batched": proposal["is_batched"],
                        "vault_total_assets": proposal["total_assets"],
                        "vault_apr_before_pct": proposal["vault_apr_before_pct"],
                        "vault_apr_after_pct": proposal["vault_apr_after_pct"],
                        "strategy": row["strategy"],
                        "name": row["name"],
                        "changed": row["changed"],
                        "ratio_before_bps": row["ratio_before_bps"],
                        "ratio_after_bps": row["ratio_after_bps"],
                        "debt_ratio_before_pct": row["debt_ratio_before_pct"],
                        "debt_ratio_after_pct": row["debt_ratio_after_pct"],
                        "current_debt": row["current_debt"],
                        "target_debt": row["target_debt"],
                        "nominal_delta": row["nominal_delta"],
                        "apr_before_1e18": row["apr_before_1e18"],
                        "apr_after_1e18": row["apr_after_1e18"],
                        "apr_before_pct": row["apr_before_pct"],
                        "apr_after_pct": row["apr_after_pct"],
                        "apr_contribution_before_pct": row["apr_contribution_before_pct"],
                        "apr_contribution_after_pct": row["apr_contribution_after_pct"],
                    }
                )

    return json_path, csv_path


def run(
    vault_address: str,
    allocator_address: str,
    applicator_address: str,
    apr_oracle_address: str,
    multicall_address: str,
    start_block: int | None,
    end_block: int | None,
    out_prefix: str | None,
    log_chunk: int,
    multicall_batch: int,
    network_name: str | None = "mainnet",
) -> Dict[str, Any]:
    if not network.is_connected():
        if network_name is None:
            raise RuntimeError(
                "brownie network is not connected. Provide network_name or run via brownie --network."
            )
        network.connect(network_name)

    vault_address = Web3.to_checksum_address(vault_address)
    allocator_address = Web3.to_checksum_address(allocator_address)
    applicator_address = Web3.to_checksum_address(applicator_address)
    apr_oracle_address = Web3.to_checksum_address(apr_oracle_address)
    multicall_address = Web3.to_checksum_address(multicall_address)

    allocator = get_contract(web3, "DebtAllocator.json", allocator_address)
    vault = get_contract(web3, "VaultV3.json", vault_address)
    apr_oracle = get_contract(web3, "AprOracle.json", apr_oracle_address)
    multicall = get_contract(web3, "Multicall.json", multicall_address)

    latest_block = web3.eth.block_number
    resolved_start = start_block
    if resolved_start is None:
        resolved_start = find_contract_deployment_block(allocator_address)
    resolved_end = latest_block if end_block is None else min(end_block, latest_block)

    if resolved_start > resolved_end:
        raise ValueError("start block must be <= end block")

    print(
        f"Scanning logs for vault={vault_address} allocator={allocator_address} blocks={resolved_start}-{resolved_end}"
    )

    grouped_logs = fetch_ratio_update_logs(
        allocator=allocator,
        vault=vault_address,
        from_block=resolved_start,
        to_block=resolved_end,
        log_chunk=log_chunk,
    )

    ordered_tx_items = sorted(
        grouped_logs.values(),
        key=lambda item: (item["block_number"], item["transaction_index"]),
    )

    print(f"Found {len(ordered_tx_items)} txs with allocator ratio updates for this vault")

    filtered_tx_items: List[Dict[str, Any]] = []
    for tx_item in ordered_tx_items:
        tx = web3.eth.get_transaction(tx_item["tx_hash"])
        tx_to = tx["to"]
        if tx_to is None:
            continue
        if Web3.to_checksum_address(tx_to) != applicator_address:
            continue
        tx_item["tx_from"] = Web3.to_checksum_address(tx["from"])
        filtered_tx_items.append(tx_item)

    print(f"Kept {len(filtered_tx_items)} txs where tx.to == applicator")

    fallback_count = 0
    proposals: List[Dict[str, Any]] = []

    try:
        vault_name = vault.functions.name().call()
    except Exception:
        vault_name = _short(vault_address)

    for index, tx_item in enumerate(filtered_tx_items, start=1):
        proposal, used_fallback = analyze_tx(
            tx_item=tx_item,
            vault=vault,
            allocator=allocator,
            apr_oracle=apr_oracle,
            multicall=multicall,
            vault_address=vault_address,
            multicall_batch=multicall_batch,
        )
        proposal["tx_from"] = tx_item.get("tx_from")
        proposals.append(proposal)
        if used_fallback:
            fallback_count += 1

        print(f"Processed {index}/{len(filtered_tx_items)}")
        print_proposal_summary(vault_name, proposal)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    if out_prefix is None:
        out_prefix = f"reports/debt_allocator_replay_{vault_address}_{timestamp}"

    metadata = {
        "chain_id": int(chain.id),
        "network": network.show_active(),
        "vault": vault_address,
        "vault_name": vault_name,
        "allocator": allocator_address,
        "applicator": applicator_address,
        "apr_oracle": apr_oracle_address,
        "multicall": multicall_address,
        "start_block": resolved_start,
        "end_block": resolved_end,
        "scanned_update_txs": len(ordered_tx_items),
        "doa_update_txs": len(filtered_tx_items),
        "fallback_used_count": fallback_count,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
    }

    json_path, csv_path = write_outputs(
        proposals=proposals,
        out_prefix=out_prefix,
        metadata=metadata,
    )

    print("\nDone.")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")

    return {
        "json_path": json_path,
        "csv_path": csv_path,
        "meta": metadata,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay DebtAllocator ratio proposals posted through DOA and reconstruct APR impact."
    )
    parser.add_argument("--vault", default=DEFAULT_VAULT)
    parser.add_argument("--allocator", default=DEFAULT_ALLOCATOR)
    parser.add_argument("--applicator", default=DEFAULT_APPLICATOR)
    parser.add_argument("--apr-oracle", default=DEFAULT_APR_ORACLE)
    parser.add_argument("--multicall", default=DEFAULT_MULTICALL3)
    parser.add_argument("--start-block", type=int, default=None)
    parser.add_argument("--end-block", type=int, default=None)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--log-chunk", type=int, default=DEFAULT_LOG_CHUNK)
    parser.add_argument("--multicall-batch", type=int, default=DEFAULT_MULTICALL_BATCH)
    parser.add_argument("--network-name", default="mainnet")
    return parser


def main(
    vault: str = DEFAULT_VAULT,
    allocator: str = DEFAULT_ALLOCATOR,
    applicator: str = DEFAULT_APPLICATOR,
    apr_oracle: str = DEFAULT_APR_ORACLE,
    multicall: str = DEFAULT_MULTICALL3,
    start_block: int | None = None,
    end_block: int | None = None,
    out_prefix: str | None = None,
    log_chunk: int = DEFAULT_LOG_CHUNK,
    multicall_batch: int = DEFAULT_MULTICALL_BATCH,
    network_name: str | None = None,
):
    return run(
        vault_address=vault,
        allocator_address=allocator,
        applicator_address=applicator,
        apr_oracle_address=apr_oracle,
        multicall_address=multicall,
        start_block=start_block,
        end_block=end_block,
        out_prefix=out_prefix,
        log_chunk=log_chunk,
        multicall_batch=multicall_batch,
        network_name=network_name,
    )


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(
        vault_address=args.vault,
        allocator_address=args.allocator,
        applicator_address=args.applicator,
        apr_oracle_address=args.apr_oracle,
        multicall_address=args.multicall,
        start_block=args.start_block,
        end_block=args.end_block,
        out_prefix=args.out_prefix,
        log_chunk=args.log_chunk,
        multicall_batch=args.multicall_batch,
        network_name=args.network_name,
    )
