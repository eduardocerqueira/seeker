#date: 2024-08-23T17:11:13Z
#url: https://api.github.com/gists/2f71636fe5061148a3b01fbc19aa506c
#owner: https://api.github.com/users/TrynD1

from binance.client import Client
from ..models.settings import BotSettings
from ..services.logging_service import log_trade
from ..routers.websockets import manager
import asyncio

async def start_trading_bot(settings: BotSettings):
    client = "**********"="your_binance_api_key", api_secret="your_binance_secret_key")

    # 레버리지 설정
    client.futures_change_leverage(symbol=settings.symbol, leverage=settings.leverage)

    # 교차/격리 거래 설정
    client.futures_change_margin_type(symbol=settings.symbol, marginType=settings.margin_type.upper())

    def check_pnl(symbol, entry_price, target_pnl):
        """ 특정 PNL 도달 여부를 확인하는 함수 """
        position = client.futures_position_information(symbol=symbol)
        for pos in position:
            if pos['entryPrice'] == entry_price:
                pnl = float(pos['unrealizedProfit'])
                if pnl >= target_pnl:
                    return True
        return False

    # 각 조건에 따라 거래 실행
    for condition in settings.conditions:
        order = None
        if settings.entry_price_type == "MARKET":
            order = client.futures_create_order(
                symbol=settings.symbol,
                side=settings.order_type,  # 진입 방향에 따라 LONG/SHORT
                type="MARKET",
                quantity=condition.collateral  # 증거금 기반으로 수량 계산 필요
            )
        elif settings.entry_price_type == "LIMIT":
            order = client.futures_create_order(
                symbol=settings.symbol,
                side=settings.order_type,
                type="LIMIT",
                price=condition.entry_price,
                quantity=condition.collateral
            )

        # 로그 기록 및 WebSocket 메시지 전송
        log_trade(f"Order placed: {order}")
        await manager.send_message(f"Order placed: {order}")

        # 익절률에 도달할 때까지 대기
        while True:
            if check_pnl(settings.symbol, order['price'], condition.pnl_target):
                # 익절 처리
                sell_order = client.futures_create_order(
                    symbol=settings.symbol,
                    side="SELL" if settings.order_type == "LONG" else "BUY",  # 익절 시 반대 방향 주문
                    type="MARKET",
                    quantity=condition.collateral
                )
                log_trade(f"Take profit: {sell_order}")
                await manager.send_message(f"Take profit: {sell_order}")
                break
            await asyncio.sleep(5)  # 비동기 함수 내에서 대기

        # 반복 실행 여부 확인
        if not settings.repeat_execution:
            break

    await manager.send_message("거래 완료")
    return "거래 완료"
"
