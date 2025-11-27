#date: 2025-11-27T16:53:23Z
#url: https://api.github.com/gists/52997957db7355e689998fc62f288117
#owner: https://api.github.com/users/ieagle2628-png

# bot.py

import time
from typing import Dict, Any, List

import MetaTrader5 as mt5

from core.config_loader import load_settings, load_symbols_json
from core.state_store import StateStore
from core.models import SymbolConstraints, SignalDecision, EntryPlan
from monitoring.logger import get_logger
from monitoring.metrics import Metrics
from mt5_io.connect import ensure_connected
from mt5_io.account import get_account_snapshot
from mt5_io.symbols import resolve_watchlist, fetch_symbol_constraints
from mt5_io.orders import order_check_send
from risk_engine.guards import can_trade_now
from risk_engine.sizing import compute_lots_each
from execution_engine.pack_manager import PackManager
from execution_engine.soft_tp import soft_tp_loop
from failsafe.dummy_stops import compute_dummy_sl_points
from signal_hub.momentum_scalp import StrategyImpl as MomentumScalp
from signal_hub.pullback_scalp import StrategyImpl as PullbackScalp

# Optional structure confidence (ensure module exists if you enable)
try:
    from structure_engine.confidence import compute_confidence
    HAS_STRUCTURE = True
except Exception:
    HAS_STRUCTURE = False


def find_symbol_meta(symbols_json, sym):
    """
    Look up metadata for a symbol inside symbols_json.
    Works whether symbols_json is a dict or a list.
    """
    if isinstance(symbols_json, dict):
        return symbols_json.get(sym, {})
    elif isinstance(symbols_json, list):
        for entry in symbols_json:
            if entry.get("symbol") == sym:
                return entry
    return {}


class Datarizer:
    def __init__(self, mt5_api):
        self.mt5 = mt5_api
        self._ohlc = {}

    def preload(self, symbols, timeframe=mt5.TIMEFRAME_M1, bars=200):
        for sym in symbols:
            rates = self.mt5.copy_rates_from_pos(sym, timeframe, 0, bars)
            if rates is None or len(rates) == 0:
                self._ohlc[sym] = []
                continue
            ohlc = [{"o": r["open"], "h": r["high"], "l": r["low"], "c": r["close"]} for r in rates]
            self._ohlc[sym] = ohlc

    def get_context(self, symbol, timeframe=mt5.TIMEFRAME_M1):
        if symbol not in self._ohlc:
            self.preload([symbol], timeframe=timeframe, bars=200)
        return {"ohlc": self._ohlc.get(symbol, [])}


class TradeSelector:
    def __init__(self, store, settings, symbols_json):
        self.store = store
        self.settings = settings
        self.symbols_json = symbols_json

    def _score_symbol(self, sym: str, ctx: dict, sc, tick_delta: float, structure_obj: dict) -> float:
        # Momentum + candle alignment
        m = momentum_signal(ctx.get("ohlc", []))
        c = candle_context(ctx.get("ohlc", []), ma_len=20)

        score = 0.0
        if m.get("ok"):
            score += 0.6 if m.get("up") or m.get("down") else 0.0
        if c.get("ok"):
            score += 0.2 if (c["two_up"] or c["two_down"]) else 0.0
            score += 0.1 if (c["engulf_up"] or c["engulf_down"]) else 0.0

        # Structure confidence bias (optional)
        if structure_obj and isinstance(structure_obj.get("score", None), (int, float)):
            # Prefer clean structure: lower noise => add score
            score += max(0.0, 0.3 - abs(0.5 - float(structure_obj["score"])) * 0.3)

        # Penalize wide spreads
        symbol_meta = find_symbol_meta(self.symbols_json, sym)
        max_spread_allowed = int(symbol_meta.get("max_spread_points", self.settings["symbols"]["max_spread_points"]))
        spread_pts = int(getattr(sc, "spread_points", 0))
        if spread_pts > max_spread_allowed:
            score -= 0.8

        return score

    def pick(self, watchlist, datarizer, acct_snapshot) -> Dict[str, Any]:
        # Configurable thresholds
        min_score = float(self.settings.get("execution", {}).get("min_entry_score", 0.8))
        cooldown_sec = float(self.settings.get("execution", {}).get("selector_cooldown_sec", 10.0))
        max_active_packs = int(self.settings.get("execution", {}).get("selector_max_active_packs", 1))
       
       
        # In soft-TP action 'closed' and in the dynamic harvest close block:
        active_packs = store.get("selector_active_packs") or 0
        store.set("selector_active_packs", max(0, active_packs - 1))

        # Respect concurrency cap
        active_packs = self.store.get("selector_active_packs") or 0
        if active_packs >= max_active_packs:
            return {"chosen": False, "reason": "max_active", "sym": None}

        best = {"sym": None, "score": -1.0, "dir": None, "ctx": None, "sc": None, "tick": None}

        for sym in watchlist:
            sc = fetch_symbol_constraints(sym)
            tick = mt5.symbol_info_tick(sym)
            if not tick or not sc:
                continue

            # Cooldown per symbol
            cd_key = f"cooldown:{sym}"
            cd = self.store.get(cd_key) or 0
            now = time.time()
            if now < cd:
                continue
    selector = TradeSelector(store=store, settings=settings, symbols_json=symbols_json)

    while True:
        acct = get_account_snapshot()
        datarizer.preload(watchlist)  # refresh context batch

        pick = selector.pick(watchlist, datarizer, acct)
        if not pick.get("chosen"):
            time.sleep(0.5)
            continue

        sym = pick["sym"]
        sc = pick["sc"]
        tick = pick["tick"]
        ctx = pick["ctx"]
        ask, bid = float(tick.ask), float(tick.bid)
        tick_delta = ask - bid

        md = {
            "symbol": sym,
            "tick_delta": tick_delta,
            "trend_dir": "buy" if tick_delta >= 0 else "sell",
            "soft_tp_points": int(settings["symbols"]["soft_tp_points"]),
            "price": ask,
            "context": ctx,
            "structure": {},  # already used in selector; optional here
        }

        # Only run strategies for the chosen symbol
        for strat in strategies:
            decision: SignalDecision = strat.decide(md, settings)
            if not decision or not getattr(decision, "should_trade", True):
                continue

            # ... sizing, build plan, send, pack_mgr.create_pack, log ...

            # Guard spread and margin on a per-symbol basis
            symbol_meta_guard = find_symbol_meta(self.symbols_json, sym)
            max_spread_allowed = int(symbol_meta_guard.get("max_spread_points", self.settings["symbols"]["max_spread_points"]))
            ok_margin, reason = can_trade_now(acct_snapshot, sc, self.settings["account"]["margin_level_floor"], max_spread_allowed)
            if not ok_margin:
                continue

            # Build context and score
            ctx = datarizer.get_context(sym, timeframe=mt5.TIMEFRAME_M1)
            structure_obj = {}
            if HAS_STRUCTURE:
                try:
                    structure_obj = compute_confidence(ctx["ohlc"])
                except Exception:
                    pass

            ask, bid = float(tick.ask), float(tick.bid)
            tick_delta = ask - bid
            s = self._score_symbol(sym, ctx, sc, tick_delta, structure_obj)

            if s > best["score"]:
                # Direction based on momentum
                m = momentum_signal(ctx["ohlc"])
                dirn = "buy" if m.get("up") else ("sell" if m.get("down") else None)
                best = {"sym": sym, "score": s, "dir": dirn, "ctx": ctx, "sc": sc, "tick": tick}

        # Enforce threshold
        if best["score"] >= min_score and best["dir"] is not None:
            # Apply cooldown
            self.store.set(f"cooldown:{best['sym']}", time.time() + cooldown_sec)
            # Increment active count
            self.store.set("selector_active_packs", active_packs + 1)
            return {"chosen": True, **best}
        return {"chosen": False, "reason": "no_symbol", "sym": None}


# --- Batch state helpers ---
# --- Batch state helpers ---
def batch_key(symbol: str, direction: str) -> str:
    return f"batch:{symbol}:{direction}"

def get_batch(store, symbol: str, direction: str):
    return store.get(batch_key(symbol, direction)) or {
        "harvested_usd": 0.0,
        "open_pnl_usd": 0.0,
        "legs": 0,
        "max_unrealized_usd": 0.0,
        "last_update": 0.0,
    }

def set_batch(store, symbol: str, direction: str, data: dict):
    store.set(batch_key(symbol, direction), data)

def compute_soft_floor_usd(symbol_meta: dict, base_floor_usd: float, lots: float) -> float:
    """
    Scale profit floor by lot and tick value if provided; otherwise return base.
    """
    tv = float(symbol_meta.get("tick_value", 0.0) or 0.0)
    ts = float(symbol_meta.get("tick_size", 0.0) or 0.0)
    override_floor = symbol_meta.get("soft_tp_min_usd")
    if override_floor is not None:
        base_floor_usd = float(override_floor)

    # If tick_value is known, anchor floor to 'a few ticks worth' scaled by lots
    if tv > 0.0 and ts > 0.0 and lots > 0.0:
        # Aim for ~150–250 ticks worth of PnL per pack as a baseline
        ticks_target = 200.0
        scaled_floor = ticks_target * tv * lots
        # Keep within reasonable bounds relative to base_floor
        return max(base_floor_usd, min(base_floor_usd * 3.0, scaled_floor))
    return base_floor_usd

# --- Momentum helper ---
def candle_context(ohlc, ma_len=20):
    if not ohlc or len(ohlc) < ma_len+2: return {"ok": False}
    closes = [b["c"] for b in ohlc]
    ma = sum(closes[-ma_len:])/ma_len
    last, prev = ohlc[-1], ohlc[-2]
    two_up = last["c"] > last["o"] and prev["c"] > prev["o"]
    two_down = last["c"] < last["o"] and prev["c"] < prev["o"]
    engulf_down = (prev["c"] > prev["o"]) and (last["o"] > prev["c"] and last["c"] < prev["o"])
    engulf_up = (prev["c"] < prev["o"]) and (last["o"] < prev["c"] and last["c"] > prev["o"])
    return {"ok": True, "ma": ma, "two_up": two_up, "two_down": two_down,
            "engulf_up": engulf_up, "engulf_down": engulf_down, "cl": closes[-1]}


def momentum_signal(ohlc: List[Dict[str, float]], fast=12, slow=26, signal=9, ma_len=50) -> Dict[str, Any]:
    closes = [bar["c"] for bar in ohlc][-max(slow + signal, ma_len):]
    if len(closes) < max(slow + signal, ma_len):
        return {"ok": False}
    def ema(vals, p):
        k = 2 / (p + 1)
        e = vals[0]
        for v in vals[1:]:
            e = v * k + e * (1 - k)
        return e
    macd_line = ema(closes, fast) - ema(closes, slow)
    signal_line = ema(closes, signal)
    ma = sum(closes[-ma_len:]) / ma_len
    momentum_up = macd_line > signal_line and closes[-1] > ma
    momentum_down = macd_line < signal_line and closes[-1] < ma
    return {"ok": True, "up": momentum_up, "down": momentum_down, "ma": ma, "cl": closes[-1]}


def load_strategies() -> List[Any]:
    """
    Load strategy implementations. You can switch to auto-discovery later.
    """
    return [MomentumScalp(), PullbackScalp()]


def build_requests(
    entry_plan: EntryPlan,
    constraints: SymbolConstraints,
    ask: float,
    bid: float,
    bot_magic: int,
    comment: str,
) -> List[Dict[str, Any]]:
    """
    Build market order requests with optional dummy SL. No broker TP (soft TP managed).
    """
    reqs: List[Dict[str, Any]] = []
    is_buy = entry_plan.direction == "buy"
    price = ask if is_buy else bid

    for _ in range(entry_plan.count):
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": entry_plan.symbol,
            "volume": entry_plan.lots_each,
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": 10,
            "magic": bot_magic,
            "comment": comment,
            "type_filling": mt5.ORDER_FILLING_FOK,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        # Broker-side SL: enforce minimum distance and precision to avoid 10016
        # if entry_plan.use_dummy_sl_tp and constraints.point > 0:
        #     pt = constraints.point
        #     min_sl_points = int(max(constraints.stops_level, constraints.freeze_level))
        #     sl_points = max(int(entry_plan.dummy_sl_points), min_sl_points)

        #     sl = price - sl_points * pt if is_buy else price + sl_points * pt
        #     factor = 10 ** constraints.digits
        #     sl = round(sl * factor) / factor

        #     if is_buy and sl >= price:
        #         sl = round((price - sl_points * pt) * factor) / factor
        #     elif (not is_buy) and sl <= price:
        #         sl = round((price + sl_points * pt) * factor) / factor

        #     req["sl"] = sl

        # No broker TP; soft TP handled by management loop
        reqs.append(req)

    return reqs


def decide_temperament(instant_mode_cfg: Any, structure_obj: Dict[str, Any]) -> bool:
    """
    Map structure confidence to instant-on-blue. If 'auto', use score threshold.
    """
    if isinstance(instant_mode_cfg, bool):
        return instant_mode_cfg
    if str(instant_mode_cfg).lower() == "auto" and structure_obj:
        return float(structure_obj.get("score", 0.3)) < 0.55
    return True  # default to instant-on-blue


def main():
    settings = load_settings("config/settings.yaml")
    symbols_json = load_symbols_json("config/symbols.json")
    logger = get_logger(settings)
    metrics = Metrics(settings)
    store = StateStore()

    ensure_connected(logger)
    acct = get_account_snapshot()

    watchlist = resolve_watchlist(settings, symbols_json, logger)
    datarizer = Datarizer(mt5)
    datarizer.preload(watchlist)

    pack_mgr = PackManager(store=store, logger=logger)
    strategies = load_strategies()

    exec_cfg = settings.get("execution", {})
    bot_magic = int(exec_cfg.get("bot_magic", 271828))
    comment = str(exec_cfg.get("order_comment", "mt5_scalper_bot"))
    instant_mode_cfg = exec_cfg.get("instant_on_blue", "auto")
    extend_if_momentum = bool(exec_cfg.get("extend_if_momentum", False))
    min_blue_default = float(exec_cfg.get("min_blue_usd", 0.01))

    while True:
        acct = get_account_snapshot()

        for sym in watchlist:
            sc = fetch_symbol_constraints(sym)
            constraints = SymbolConstraints(
                symbol=sym,
                point=sc.point,
                digits=sc.digits,
                volume_min=sc.volume_min,
                volume_max=sc.volume_max,
                volume_step=sc.volume_step,
                spread_points=sc.spread_points,
                stops_level=sc.stops_level,
                freeze_level=sc.freeze_level,
                contract_size=getattr(sc, "contract_size", 0.0),
                tick_value=getattr(sc, "tick_value", 0.0),
                value_per_point=getattr(sc, "value_per_point", None),
            )

            def momentum_signal(ohlc: List[Dict[str, float]], fast=12, slow=26, signal=9, ma_len=50) -> Dict[str, Any]:
                closes = [bar["c"] for bar in ohlc][-max(slow + signal, ma_len):]
                if len(closes) < max(slow + signal, ma_len):
                    return {"ok": False}
                def ema(vals, p):
                    k = 2 / (p + 1)
                    e = vals[0]
                    for v in vals[1:]:
                        e = v * k + e * (1 - k)
                    return e
                macd_line = ema(closes, fast) - ema(closes, slow)
                signal_line = ema(closes, signal)
                ma = sum(closes[-ma_len:]) / ma_len
                momentum_up = macd_line > signal_line and closes[-1] > ma
                momentum_down = macd_line < signal_line and closes[-1] < ma
                return {"ok": True, "up": momentum_up, "down": momentum_down, "ma": ma, "cl": closes[-1]}


            # Risk/margin guard
          # Risk/margin guard (use per-symbol spread limit if present)
                symbol_meta_guard = find_symbol_meta(symbols_json, sym)
                max_spread_allowed = int(symbol_meta_guard.get("max_spread_points", settings["symbols"]["max_spread_points"]))
                ok_margin, reason = can_trade_now(acct, sc, settings["account"]["margin_level_floor"], max_spread_allowed)
                
                # symbol_meta_guard = find_symbol_meta(symbols_json, sym)
                # max_spread_allowed = int(symbol_meta_guard.get("max_spread_points", settings["symbols"]["max_spread_points"]))

                # ok_margin, reason = can_trade_now(
                #     acct,
                #     sc,
                #     settings["account"]["margin_level_floor"],
                #     max_spread_allowed,
                # )
                # if not ok_margin:
                logger.debug(f"[GUARD BLOCK] {sym}: {reason}")
                return

            # Tick and prices
            tick = mt5.symbol_info_tick(sym)
            if not tick:
                logger.debug(f"[NO TICK] {sym}")
                continue
            ask, bid = float(tick.ask), float(tick.bid)
            tick_delta = ask - bid

            # Context and optional structure confidence
            ctx = datarizer.get_context(sym, timeframe=mt5.TIMEFRAME_M1)
            structure_obj = {}
            if HAS_STRUCTURE:
                try:
                    structure_obj = compute_confidence(ctx["ohlc"])
                except Exception as e:
                    logger.debug(f"[STRUCTURE ERR] {sym}: {e}")

            store.set(f"struct_{sym}", structure_obj)

            # Strategy decisions
            # Strategy decisions
            md = {
                "symbol": sym,
                "tick_delta": tick_delta,
                "trend_dir": "buy" if tick_delta >= 0 else "sell",
                "soft_tp_points": int(settings["symbols"]["soft_tp_points"]),
                "price": ask,
                "context": ctx,
                "structure": structure_obj,
            }

            symbol_meta = find_symbol_meta(symbols_json, sym)

            # Momentum seed (optional anchor), independent of the strategy loop
            m = momentum_signal(ctx["ohlc"])
            if m.get("ok"):
                seed_dir = "buy" if m.get("up") else ("sell" if m.get("down") else None)
                if seed_dir:
                    existing = any(
                        (pack := pack_mgr.get_pack(pid)) and pack.symbol == sym and pack.direction == seed_dir
                        for pid in pack_mgr.list_packs()
                    )
                    if not existing:
                        risk_cap_seed = float(symbol_meta.get("risk_dollars_cap_micro", settings["account"].get("risk_dollars_cap_micro", 0.02)))
                        base_sl_points_seed = compute_dummy_sl_points(sym, sc)
                        lots_each_seed = compute_lots_each(
                            constraints=constraints,
                            risk_dollars=risk_cap_seed,
                            sl_points=base_sl_points_seed,
                            count=1,
                            symbol_meta=symbol_meta,
                            account_balance=acct.balance,
                        )
                        if lots_each_seed and lots_each_seed > 0.0:
                            plan_seed = EntryPlan(
                                symbol=sym,
                                direction=seed_dir,
                                lots_each=lots_each_seed,
                                count=1,
                                use_dummy_sl_tp=True,
                                dummy_sl_points=base_sl_points_seed,
                                dummy_tp_points=int(settings["symbols"]["soft_tp_points"]),
                            )
                            reqs_seed = build_requests(plan_seed, constraints, ask, bid, bot_magic, comment)
                            tickets_seed = order_check_send(reqs_seed, logger)
                            if tickets_seed:
                                pack_mgr.create_pack(
                                    symbol=sym,
                                    direction=seed_dir,
                                    ticket_ids=tickets_seed,
                                    soft_tp_points=int(settings["symbols"]["soft_tp_points"]),
                                    soft_timeout_sec=15,
                                    strategy_name="MomentumSeed",
                                )
                                logger.info(f"[SEED] {sym} {seed_dir} lots_each={lots_each_seed}")

            # Main strategies — build plan, send, create pack inside the loop
            for strat in strategies:
                decision: SignalDecision = strat.decide(md, settings)
                if not decision or not getattr(decision, "should_trade", True):
                    continue

                # Per-symbol risk override for high-vol instruments
                risk_cap = float(symbol_meta.get("risk_dollars_cap_micro", settings["account"].get("risk_dollars_cap_micro", 0.02)))

                # SL points from strategy decision or fallback dummy
                if getattr(decision, "sl_points", None) is not None:
                    base_sl_points = int(decision.sl_points)
                else:
                    base_sl_points = compute_dummy_sl_points(sym, sc)

                lots_each = compute_lots_each(
                    constraints=constraints,
                    risk_dollars=risk_cap,
                    sl_points=base_sl_points,
                    count=decision.count,
                    symbol_meta=symbol_meta,
                    account_balance=acct.balance,
                )
                if not lots_each or lots_each <= 0.0:
                    logger.debug(f"[SIZE SKIP] {sym} risk too high for min volume (sl={base_sl_points})")
                    continue

                plan = EntryPlan(
                    symbol=sym,
                    direction=decision.direction,
                    lots_each=lots_each,
                    count=decision.count,
                    use_dummy_sl_tp=True,
                    dummy_sl_points=base_sl_points,
                    dummy_tp_points=int(md["soft_tp_points"]),
                )

                reqs = build_requests(
                    entry_plan=plan,
                    constraints=constraints,
                    ask=ask,
                    bid=bid,
                    bot_magic=bot_magic,
                    comment=comment,
                )
                tickets = order_check_send(reqs, logger)
                if not tickets:
                    logger.debug(f"[SEND FAILED] {sym} {plan.direction}")
                    continue

                soft_timeout_sec = int(getattr(decision, "soft_timeout_sec", 15))
                pack_mgr.create_pack(
                    symbol=sym,
                    direction=plan.direction,
                    ticket_ids=tickets,
                    soft_tp_points=int(md["soft_tp_points"]),
                    soft_timeout_sec=soft_timeout_sec,
                    strategy_name=getattr(strat, "name", "Unknown"),
                )

                fmt_price = f"{ask:.{constraints.digits}f}"
                logger.info(
                    f"[ENTRY] {sym} dir={plan.direction} lots_each={plan.lots_each} "
                    f"count={plan.count} sl_pts={plan.dummy_sl_points} price={fmt_price}"
                )


        # Register manual trades once (no duplicates), then optionally scale them
        positions = mt5.positions_get()
        if positions:
            managed_tickets = set()
            for pack_id in pack_mgr.list_packs():
                pack = pack_mgr.get_pack(pack_id)
                if pack:
                    managed_tickets.update(pack.ticket_ids)

            for pos in positions:
                if pos.symbol not in watchlist:
                    continue
                if pos.ticket in managed_tickets:
                    continue  # already tracked

                pack_mgr.create_pack(
                    symbol=pos.symbol,
                    direction="buy" if pos.type == mt5.ORDER_TYPE_BUY else "sell",
                    ticket_ids=[pos.ticket],
                    soft_tp_points=int(settings["symbols"]["soft_tp_points"]),
                    soft_timeout_sec=15,
                    strategy_name="ManualFollow",
                )
                logger.info(f"[FOLLOW] Registered manual {pos.symbol} ticket={pos.ticket}")
                managed_tickets.add(pos.ticket)
                # --- Manual follow-through scaling (bit-by-bit) ---
                extend_count_max = int(settings["execution"].get("manual_extend_count", 3))
                extend_delay_sec = float(settings["execution"].get("manual_extend_delay_sec", 2.0))
                extend_price_step_pts = int(settings["execution"].get("manual_extend_price_step_pts", 5))

                # If scaling is disabled, skip
                if extend_count_max <= 0:
                    continue

                # Scheduler state per manual ticket
                sched_key = f"extend_sched:{pos.symbol}:{pos.ticket}"
                sched = store.get(sched_key) or {"added": 0, "last_price": None, "last_time": 0}

                # Fresh constraints/ticks for this symbol
                sc_ext = fetch_symbol_constraints(pos.symbol)
                constraints_ext = SymbolConstraints(
                    symbol=pos.symbol,
                    point=sc_ext.point,
                    digits=sc_ext.digits,
                    volume_min=sc_ext.volume_min,
                    volume_max=sc_ext.volume_max,
                    volume_step=sc_ext.volume_step,
                    spread_points=sc_ext.spread_points,
                    stops_level=sc_ext.stops_level,
                    freeze_level=sc_ext.freeze_level,
                    contract_size=getattr(sc_ext, "contract_size", 0.0),
                    tick_value=getattr(sc_ext, "tick_value", 0.0),
                    value_per_point=getattr(sc_ext, "value_per_point", None),
                )

                tick_ext = mt5.symbol_info_tick(pos.symbol)
                if not tick_ext:
                    continue
                # Follow the same side as the manual position
                manual_dir = "buy" if pos.type == mt5.ORDER_TYPE_BUY else "sell"
                ask_ext, bid_ext = float(tick_ext.ask), float(tick_ext.bid)
                price_now = ask_ext if manual_dir == "buy" else bid_ext
                now = time.time()

                # Decide if we should add a leg
                should_add = (
                    sched["added"] < extend_count_max and
                    (now - sched["last_time"] >= extend_delay_sec) and
                    (
                        sched["last_price"] is None or
                        abs(price_now - float(sched["last_price"])) >= extend_price_step_pts * constraints_ext.point
                    )
                )

                if should_add:
                    # Small, safe sizing
                    risk_cap_ext = float(settings["account"].get("risk_dollars_cap_micro", 0.02))
                    base_sl_points_ext = compute_dummy_sl_points(pos.symbol, sc_ext)
                    symbol_meta_ext = find_symbol_meta(symbols_json, pos.symbol)

                    lots_each_ext = compute_lots_each(
                        constraints=constraints_ext,
                        risk_dollars=risk_cap_ext,
                        sl_points=base_sl_points_ext,
                        count=1,
                        symbol_meta=symbol_meta_ext,
                        account_balance=acct.balance,
                    )

                    # Snap to broker step and clamp to min/max
                    def snap_volume(vol, step, vmin, vmax):
                        if step and step > 0:
                            vol = (int(vol / step)) * step
                        return max(vmin, min(vol, vmax))

                    if lots_each_ext and lots_each_ext > 0.0:
                        lots_each_ext = snap_volume(
                            lots_each_ext, constraints_ext.volume_step,
                            constraints_ext.volume_min, constraints_ext.volume_max
                        )

                    if not lots_each_ext or lots_each_ext < constraints_ext.volume_min:
                        logger.debug(f"[EXTEND SKIP] {pos.symbol} snapped volume too small ({lots_each_ext})")
                    else:
                        plan_ext = EntryPlan(
                            symbol=pos.symbol,
                            direction=manual_dir,
                            lots_each=lots_each_ext,
                            count=1,
                            use_dummy_sl_tp=True,
                            dummy_sl_points=base_sl_points_ext,
                            dummy_tp_points=int(settings["symbols"]["soft_tp_points"]),
                        )

                        reqs_ext = build_requests(
                            entry_plan=plan_ext,
                            constraints=constraints_ext,
                            ask=ask_ext,
                            bid=bid_ext,
                            bot_magic=bot_magic,
                            comment=comment,
                        )
                        tickets_ext = order_check_send(reqs_ext, logger)
                        if tickets_ext:
                            pack_mgr.create_pack(
                                symbol=pos.symbol,
                                direction=plan_ext.direction,
                                ticket_ids=tickets_ext,
                                soft_tp_points=int(settings["symbols"]["soft_tp_points"]),
                                soft_timeout_sec=int(settings.get("execution", {}).get("extend_soft_timeout_sec", 15)),
                                strategy_name="ManualExtend",
                            )
                            logger.info(
                                f"[EXTEND] Added 1 leg on {pos.symbol} dir={plan_ext.direction} lots_each={plan_ext.lots_each}"
                            )
                            sched["added"] += 1
                            sched["last_price"] = price_now
                            sched["last_time"] = now
                            store.set(sched_key, sched)


                # --- Manual follow-through scaling ---
                extend_count_max = int(settings["execution"].get("manual_extend_count", 3))
                extend_delay_sec = float(settings["execution"].get("manual_extend_delay_sec", 2.0))
                extend_price_step_pts = int(settings["execution"].get("manual_extend_price_step_pts", 5))

                key = f"extend_sched:{pos.symbol}:{pos.ticket}"
                sched = store.get(key) or {"added": 0, "last_price": None, "last_time": 0}

                tick_ext = mt5.symbol_info_tick(pos.symbol)
                if tick_ext:
                    price_now = float(tick_ext.ask if pos.type == mt5.ORDER_TYPE_BUY else tick_ext.bid)
                    now = time.time()

                    should_add = (
                        sched["added"] < extend_count_max and
                        (now - sched["last_time"] >= extend_delay_sec) and
                        (sched["last_price"] is None or abs(price_now - sched["last_price"]) >= extend_price_step_pts * constraints_ext.point)
                    )

                    if should_add:
                        lots_each_ext = compute_lots_each(
                            constraints=constraints_ext,
                            risk_dollars=risk_cap_ext,
                            sl_points=base_sl_points_ext,
                            count=1,
                            symbol_meta=symbol_meta_ext,
                            account_balance=acct.balance,
                        )
                        if lots_each_ext and lots_each_ext > 0.0:
                            plan_ext = EntryPlan(
                                symbol=pos.symbol,
                                direction="buy" if pos.type == mt5.ORDER_TYPE_BUY else "sell",
                                lots_each=lots_each_ext,
                                count=1,
                                use_dummy_sl_tp=True,
                                dummy_sl_points=base_sl_points_ext,
                                dummy_tp_points=int(settings["symbols"]["soft_tp_points"]),
                            )
                            reqs_ext = build_requests(plan_ext, constraints_ext, ask_ext, bid_ext, bot_magic, comment)
                            tickets_ext = order_check_send(reqs_ext, logger)
                            if tickets_ext:
                                pack_mgr.create_pack(
                                    symbol=pos.symbol,
                                    direction=plan_ext.direction,
                                    ticket_ids=tickets_ext,
                                    soft_tp_points=int(settings["symbols"]["soft_tp_points"]),
                                    soft_timeout_sec=int(settings.get("execution", {}).get("extend_soft_timeout_sec", 15)),
                                    strategy_name="ManualExtend",
                                )
                                logger.info(f"[EXTEND] Added leg on {pos.symbol} dir={plan_ext.direction} lots_each={plan_ext.lots_each}")
                                sched["added"] += 1
                                sched["last_price"] = price_now
                                sched["last_time"] = now
                                store.set(key, sched)


                # Aggressive scaling for manual trades (configurable)
                extend_count = int(settings.get("execution", {}).get("manual_extend_count", 0))
                if extend_count <= 0:
                    continue  # scaling disabled

                sc_ext = fetch_symbol_constraints(pos.symbol)
                constraints_ext = SymbolConstraints(
                    symbol=pos.symbol,
                    point=sc_ext.point,
                    digits=sc_ext.digits,
                    volume_min=sc_ext.volume_min,
                    volume_max=sc_ext.volume_max,
                    volume_step=sc_ext.volume_step,
                    spread_points=sc_ext.spread_points,
                    stops_level=sc_ext.stops_level,
                    freeze_level=sc_ext.freeze_level,
                    contract_size=getattr(sc_ext, "contract_size", 0.0),
                    tick_value=getattr(sc_ext, "tick_value", 0.0),
                    value_per_point=getattr(sc_ext, "value_per_point", None),
                )

                tick_ext = mt5.symbol_info_tick(pos.symbol)
                if not tick_ext:
                    continue
                ask_ext, bid_ext = float(tick_ext.ask), float(tick_ext.bid)

                risk_cap_ext = float(settings["account"].get("risk_dollars_cap_micro", 0.10))
                base_sl_points_ext = compute_dummy_sl_points(pos.symbol, sc_ext)
                symbol_meta_ext = find_symbol_meta(symbols_json, pos.symbol)

                lots_each_ext = compute_lots_each(
                    constraints=constraints_ext,
                    risk_dollars=risk_cap_ext,
                    sl_points=base_sl_points_ext,
                    count=extend_count,
                    symbol_meta=symbol_meta_ext,
                    account_balance=acct.balance,
                )
                if not lots_each_ext or lots_each_ext <= 0.0:
                    logger.debug(f"[EXTEND SKIP] {pos.symbol} sizing too small (sl={base_sl_points_ext})")
                    continue

                plan_ext = EntryPlan(
                    symbol=pos.symbol,
                    direction="buy" if pos.type == mt5.ORDER_TYPE_BUY else "sell",
                    lots_each=lots_each_ext,
                    count=extend_count,
                    use_dummy_sl_tp=True,
                    dummy_sl_points=base_sl_points_ext,
                    dummy_tp_points=int(settings["symbols"]["soft_tp_points"]),
                )

                reqs_ext = build_requests(
                    entry_plan=plan_ext,
                    constraints=constraints_ext,
                    ask=ask_ext,
                    bid=bid_ext,
                    bot_magic=bot_magic,
                    comment=comment,
                )
                tickets_ext = order_check_send(reqs_ext, logger)
                if not tickets_ext:
                    logger.debug(f"[EXTEND SEND FAILED] {pos.symbol} {plan_ext.direction}")
                    continue

                pack_mgr.create_pack(
                    symbol=pos.symbol,
                    direction=plan_ext.direction,
                    ticket_ids=tickets_ext,
                    soft_tp_points=int(settings["symbols"]["soft_tp_points"]),
                    soft_timeout_sec=int(settings.get("execution", {}).get("extend_soft_timeout_sec", 15)),
                    strategy_name="ManualExtend",
                )
                logger.info(
                    f"[EXTEND] Added {extend_count} legs on {pos.symbol} "
                    f"dir={plan_ext.direction} lots_each={plan_ext.lots_each}"
                )

               # Management: soft TP loop
       # Management: soft TP loop
        packs = pack_mgr.list_packs()
        instant_on_blue_global = decide_temperament(instant_mode_cfg, {})
        for action, pack_id, pnl in soft_tp_loop(
            packs=packs,
            logger=get_logger(settings),
            min_positive=min_blue_default,
            extend_if_momentum=extend_if_momentum,
            instant_on_blue=instant_on_blue_global,
        ):
            pack = pack_mgr.get_pack(pack_id)
            if not pack:
                continue

            # Per-symbol stats
            positions_live = mt5.positions_get() or []
            legs_live = [p for p in positions_live if p.symbol == pack.symbol]
            open_pnl_usd = sum(p.profit for p in legs_live) if legs_live else 0.0
            legs_count = len(legs_live)

            # Volatility proxy
            sc_ext = fetch_symbol_constraints(pack.symbol)
            vol_pts = sc_ext.spread_points if sc_ext.spread_points else 0
            vol_bonus_usd = min(2.50, vol_pts / 220.0)  # tune
            # Speed proxy: compare current PnL to last
            batch = get_batch(store, pack.symbol, pack.direction)
            speed_usd = (open_pnl_usd - batch["open_pnl_usd"])
            speed_bonus_usd = max(0.0, min(2.0, speed_usd * 0.8))  # tune

            # Update batch highs
            # Update batch highs
            max_unrealized_usd = max(batch["max_unrealized_usd"], open_pnl_usd)
            now_ts = time.time()
            stall_seconds = (now_ts - batch["last_update"]) if batch["last_update"] else 0.0

            batch.update({
                "open_pnl_usd": open_pnl_usd,
                "legs": legs_count,
                "max_unrealized_usd": max_unrealized_usd,
                "last_update": now_ts,
            })
            set_batch(store, pack.symbol, pack.direction, batch)

            # Volatility and speed bonuses
            vol_bonus_usd = min(2.50, vol_pts / 220.0)
            speed_bonus_usd = max(0.0, min(2.00, speed_usd * 0.8))

            # Stall decay
            age_decay = 0.0
            if stall_seconds > 30:
                age_decay = min(0.60, stall_seconds / 90.0)

         
            # Drawdown governance
            risk_cfg = settings.get("risk", {})
            max_legs = int(risk_cfg.get("max_legs_per_symbol", 8))
            max_dd_usd = float(risk_cfg.get("max_drawdown_usd_per_symbol", 25.0))
            dd_usd = open_pnl_usd  # negative when in loss

            # Base lock
          

            # Per-symbol live state
            positions_live = mt5.positions_get() or []
            legs_live = [p for p in positions_live if p.symbol == pack.symbol]
            open_pnl_usd = sum(p.profit for p in legs_live) if legs_live else 0.0
            legs_count = len(legs_live)

            # Constraints and symbol overrides
            sc_ext = fetch_symbol_constraints(pack.symbol)
            symbol_meta = find_symbol_meta(symbols_json, pack.symbol)

            # --- FIX: define base_lock here ---
            lots_total = sum(getattr(p, "volume", 0.0) for p in legs_live) or 1.0
            base_lock = compute_soft_floor_usd(
                symbol_meta=symbol_meta,
                base_floor_usd=float(settings["symbols"].get("soft_tp_min_usd", 2.50)),
                lots=lots_total
            )

            # Trailing lock off max unrealized
            trail_ratio = 0.80  # lock 25% of max unrealized
            trail_lock = max(0.0, max_unrealized_usd * trail_ratio)

            # Final dynamic lock
            dynamic_lock = max(
                base_lock,
                trail_lock + vol_bonus_usd + speed_bonus_usd - age_decay,
            )

            # Actions
            if action == "closed":
                pack_mgr.bulk_close_positions(pack)
                # tally harvest
                harvested = max(0.0, pnl)
                batch["harvested_usd"] += harvested
                set_batch(store, pack.symbol, pack.direction, batch)
                pack_mgr.remove_pack(pack_id)

            elif action == "extend":
                # Only extend if not overexposed and not in heavy drawdown
                if legs_count >= max_legs or dd_usd <= -max_dd_usd:
                    logger.debug(f"[EXTEND BLOCK] {pack.symbol} legs={legs_count} dd={dd_usd}")
                else:
                    tick_ext = mt5.symbol_info_tick(pack.symbol)
                    if tick_ext:
                        ask_ext, bid_ext = float(tick_ext.ask), float(tick_ext.bid)
                        risk_cap_ext = float(settings["account"].get("risk_dollars_cap_micro", 0.02))
                        sl_pts = compute_dummy_sl_points(pack.symbol, sc_ext)
                        lots_each_ext = compute_lots_each(
                            constraints=SymbolConstraints(
                                symbol=pack.symbol, point=sc_ext.point, digits=sc_ext.digits,
                                volume_min=sc_ext.volume_min, volume_max=sc_ext.volume_max,
                                volume_step=sc_ext.volume_step, spread_points=sc_ext.spread_points,
                                stops_level=sc_ext.stops_level, freeze_level=sc_ext.freeze_level,
                                contract_size=getattr(sc_ext, "contract_size", 0.0),
                                tick_value=getattr(sc_ext, "tick_value", 0.0),
                                value_per_point=getattr(sc_ext, "value_per_point", None),
                            ), # type: ignore
                            risk_dollars=risk_cap_ext, sl_points=sl_pts, count=1,
                            symbol_meta=find_symbol_meta(symbols_json, pack.symbol),
                            account_balance=acct.balance,
                        )
                        if lots_each_ext and lots_each_ext > 0.0:
                            reqs_ext = build_requests(
                                EntryPlan(
                                    symbol=pack.symbol, direction=pack.direction,
                                    lots_each=lots_each_ext, count=1,
                                    use_dummy_sl_tp=True, dummy_sl_points=sl_pts,
                                    dummy_tp_points=int(settings["symbols"]["soft_tp_points"]),
                                ),
                                SymbolConstraints(
                                    symbol=pack.symbol, point=sc_ext.point, digits=sc_ext.digits,
                                    volume_min=sc_ext.volume_min, volume_max=sc_ext.volume_max,
                                    volume_step=sc_ext.volume_step, spread_points=sc_ext.spread_points,
                                    stops_level=sc_ext.stops_level, freeze_level=sc_ext.freeze_level,
                                    contract_size=getattr(sc_ext, "contract_size", 0.0),
                                    tick_value=getattr(sc_ext, "tick_value", 0.0),
                                    value_per_point=getattr(sc_ext, "value_per_point", None),
                                ),
                                ask_ext, bid_ext, bot_magic, comment
                            )
                            tickets_ext = order_check_send(reqs_ext, logger)
                            if tickets_ext:
                                pack_mgr.create_pack(
                                    symbol=pack.symbol, direction=pack.direction,
                                    ticket_ids=tickets_ext,
                                    soft_tp_points=int(settings["symbols"]["soft_tp_points"]),
                                    soft_timeout_sec=int(settings.get("execution", {}).get("extend_soft_timeout_sec", 15)),
                                    strategy_name="BatchExtend",
                                )
                                logger.info(f"[BATCH EXTEND] {pack.symbol} dir={pack.direction} lots_each={lots_each_ext}")

            elif action == "remove":
                pack_mgr.remove_pack(pack_id)

            # Dynamic harvest trigger (outside soft_tp_loop’s internal min_positive):
            # If open PnL retraces to dynamic_lock from the high, close the pack quickly
            if max_unrealized_usd > 0 and open_pnl_usd <= dynamic_lock:
                logger.info(f"[DYN LOCK CLOSE] {pack.symbol} lock={dynamic_lock:.2f} open_pnl={open_pnl_usd:.2f}")
                pack_mgr.bulk_close_positions(pack)
                pack_mgr.remove_pack(pack_id)


        time.sleep(0.5)


if __name__ == "__main__":
    main()
