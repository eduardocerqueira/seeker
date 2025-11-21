#date: 2025-11-21T16:47:09Z
#url: https://api.github.com/gists/1cca9d2d0fccb338f95cb883ba7cb921
#owner: https://api.github.com/users/luckman538

# trading_executor_v2.py - 실전 매매 최적화 버전
import logging
import time
import os
import threading
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
from typing import Dict, Optional, List
from datetime import datetime

# 환경변수 로드
load_dotenv()

class ExchangeInterface:
    """거래소 인터페이스 추상화 클래스"""
    
    def __init__(self, api_key: "**********": str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = "**********"
        self.testnet = testnet
        self.logger = logging.getLogger(__name__)
    
    def get_futures_balance(self) -> float:
        """거래소별 USDT 선물 잔고 조회"""
        raise NotImplementedError("서브클래스에서 구현해야 합니다")
    
    def has_open_position(self, symbol: str) -> bool:
        """포지션 존재 여부 확인"""
        raise NotImplementedError("서브클래스에서 구현해야 합니다")
    
    def safe_has_open_position(self, symbol: str, timeout: int = 10) -> bool:
        """안전한 포지션 확인 (타임아웃 방지)"""
        raise NotImplementedError("서브클래스에서 구현해야 합니다")

class BinanceExchange(ExchangeInterface):
    """Binance 거래소 구현 - 실전 매매 최적화"""
    
    def __init__(self, api_key: "**********": str, testnet: bool = False):
        super().__init__(api_key, api_secret, testnet)
        self._init_client()
        self._force_robust_time_sync()
        self._cached_balance = None
        self._balance_cache_time = 0
    
    def _init_client(self):
        """Binance 클라이언트 초기화 - API 키 오류 해결"""
        try:
            if self.testnet:
                self.client = "**********"=True)
                self.client.API_URL = 'https://testnet.binancefuture.com'
                self.logger.info("✅ Testnet 클라이언트 초기화 완료")
            else:
                # 🔥 Mainnet 클라이언트 - API 키 검증 강화
                self.client = "**********"
                # API 키 즉시 검증
                try:
                    server_time = self.client.get_server_time()
                    self.logger.info("✅ Mainnet 클라이언트 초기화 및 API 키 검증 완료")
                except Exception as e:
                    if "Invalid API-key" in str(e):
                        self.logger.error("❌ Mainnet API 키 인증 실패")
                        raise Exception("API 키 인증 실패 - 실전 거래 불가")
                    else:
                        raise e
                    
        except Exception as e:
            self.logger.error(f"❌ 클라이언트 초기화 실패: {e}")
            raise
    
    def _force_robust_time_sync(self):
        """Binance 시간 동기화"""
        self.logger.info("🕒 Binance 시간 동기화 시작...")
        
        offsets = []
        for i in range(5):
            try:
                server_time = self.client.get_server_time()
                local_time = int(time.time() * 1000)
                offset = server_time['serverTime'] - local_time
                offsets.append(offset)
                
                if i == 0:
                    best_offset = offset
                elif abs(offset) < abs(best_offset):
                    best_offset = offset
                    
                self.logger.info(f"   동기화 {i+1}: 오프셋 {offset}ms")
                time.sleep(0.2)
                
            except Exception as e:
                self.logger.warning(f"   동기화 {i+1} 실패: {e}")
                time.sleep(0.5)
        
        if offsets:
            self.time_offset = min(offsets, key=abs)
            self.last_sync_time = time.time()
            self.logger.info(f"✅ Binance 최종 시간 오프셋: {self.time_offset}ms")
        else:
            self.logger.error("❌ Binance 시간 동기화 실패")
            self.time_offset = 0

    def get_futures_balance(self) -> float:
        """Binance Futures 잔고 조회 (캐싱 적용)"""
        current_time = time.time()
        if self._cached_balance is not None and (current_time - self._balance_cache_time) < 30:
            return self._cached_balance
            
        try:
            account = self.client.futures_account()
            
            total_wallet_balance = 0.0
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    total_wallet_balance = float(asset['walletBalance'])
                    break
            
            self._cached_balance = total_wallet_balance
            self._balance_cache_time = current_time
            
            self.logger.info(f"💰 Binance Futures 잔고: ${total_wallet_balance:.2f}")
            return total_wallet_balance
            
        except Exception as e:
            self.logger.error(f"❌ Binance 잔고 조회 실패: {e}")
            return 0.0

    def has_open_position(self, symbol: str) -> bool:
        """Binance 포지션 확인 - 개선된 버전"""
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            
            for position in positions:
                position_amt = float(position.get('positionAmt', 0))
                if abs(position_amt) > 0.00001:
                    self.logger.info(f"📊 Binance {symbol} 포지션: {position_amt:.6f}주")
                    return True
                        
            return False
            
        except BinanceAPIException as e:
            if e.code == -1007:
                self.logger.warning(f"⏰ {symbol} API 타임아웃: {e.message}")
            else:
                self.logger.warning(f"⚠️ {symbol} API 오류 ({e.code}): {e.message}")
            raise
        except Exception as e:
            self.logger.warning(f"⚠️ Binance 포지션 확인 실패: {e}")
            raise

    def safe_has_open_position(self, symbol: str, timeout: int = 20, max_retries: int = 2) -> bool:
        """안전한 포지션 확인 - API 오류 시 재시도 및 폴백"""
        
        for retry in range(max_retries):
            result = [None]
            exception = [None]
            
            def check_position():
                try:
                    result[0] = self.has_open_position(symbol)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=check_position)
            thread.daemon = True
            thread.start()
            
            current_timeout = timeout if retry == 0 else timeout // 2
            thread.join(current_timeout)
            
            if thread.is_alive():
                if retry < max_retries - 1:
                    self.logger.warning(f"⏰ {symbol} 포지션 확인 타임아웃, 재시도 중... ({retry+1}/{max_retries})")
                    time.sleep(0.5)
                    continue
                else:
                    self.logger.error(f"❌ {symbol} 포지션 확인 최종 타임아웃")
                    return False
            
            if exception[0]:
                error_msg = str(exception[0])
                
                # 🔥 API 키 오류 시 즉시 실패 반환 (재시도 의미 없음)
                if "Invalid API-key" in error_msg:
                    self.logger.error(f"❌ API 키 오류로 포지션 확인 불가: {symbol}")
                    return False
                    
                if 'timeout' in error_msg.lower() and retry < max_retries - 1:
                    self.logger.warning(f"⚠️ {symbol} 타임아웃 오류, 재시도 중... ({retry+1}/{max_retries})")
                    time.sleep(0.5)
                    continue
                else:
                    self.logger.warning(f"⚠️ {symbol} 포지션 확인 실패: {exception[0]}")
                    return False
            
            if result[0] is not None:
                return result[0]
    
        self.logger.error(f"❌ {symbol} 포지션 확인 완전 실패")
        return False
    
    def set_leverage(self, symbol: str, leverage: int = 20):
        """레버리지 설정"""
        try:
            result = self.client.futures_change_leverage(
                symbol=symbol,
                leverage=leverage
            )
            self.logger.info(f"✅ {symbol} 레버리지 {leverage}배 설정 완료")
            return result
        except Exception as e:
            self.logger.error(f"❌ {symbol} 레버리지 설정 실패: {e}")
            return None

    def get_symbol_info(self, symbol: str) -> Dict:
        """심볼 정보 조회"""
        try:
            exchange_info = self.client.futures_exchange_info()
            for sym_info in exchange_info['symbols']:
                if sym_info['symbol'] == symbol:
                    return sym_info
            return {}
        except Exception as e:
            self.logger.error(f"❌ {symbol} 정보 조회 실패: {e}")
            return {}

    def calculate_position_size(self, symbol: str, risk_amount: float, entry_price: float) -> float:
        """포지션 사이즈 계산 - 실제 거래소 규칙 준수"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            
            # 기본값 (대부분의 USDT 페어)
            min_qty = 0.001
            step_size = 0.001
            min_notional = 10.0
            
            if symbol_info:
                filters = symbol_info.get('filters', [])
                for filter_item in filters:
                    if filter_item['filterType'] == 'LOT_SIZE':
                        min_qty = float(filter_item['minQty'])
                        step_size = float(filter_item['stepSize'])
                    elif filter_item['filterType'] == 'MIN_NOTIONAL':
                        min_notional = float(filter_item.get('notional', 10.0))
            
            # 기본 수량 계산
            base_quantity = risk_amount / entry_price
            
            # 최소 수량 확인
            if base_quantity < min_qty:
                base_quantity = min_qty
            
            # 스텝 사이즈 적용
            if step_size > 0:
                base_quantity = (base_quantity // step_size) * step_size
                base_quantity = round(base_quantity, 8)
            
            # 최소 주문 금액 확인
            notional_value = base_quantity * entry_price
            if notional_value < min_notional:
                required_quantity = (min_notional * 1.1) / entry_price
                if step_size > 0:
                    required_quantity = (required_quantity // step_size) * step_size
                base_quantity = max(base_quantity, required_quantity)
            
            return round(base_quantity, 6)
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 포지션 계산 실패: {e}")
            # 기본 폴백
            base_quantity = risk_amount / entry_price
            return round(max(base_quantity, 10.0 / entry_price), 6)

    def robust_market_order(self, symbol: str, side: str, quantity: float, max_retries: int = 3) -> Dict:
        """강력한 시장가 주문 실행 - 실전 매매용 (구조 개선)"""
        for attempt in range(max_retries):
            try:
                self.logger.info(f"🎯 주문 시도 {attempt+1}/{max_retries}: {symbol} {side} {quantity:.6f}")
                
                # 레버리지 설정
                try:
                    self.set_leverage(symbol, 20)
                except Exception as leverage_error:
                    self.logger.warning(f"⚠️ 레버리지 설정 실패: {leverage_error}")

                # 🔥 주문 수량 최종 검증 및 반올림
                final_quantity = round(quantity, 6)
                
                # 최소 주문 금액 재확인
                symbol_info = self.get_symbol_info(symbol)
                min_notional = 10.0
                if symbol_info:
                    filters = symbol_info.get('filters', [])
                    for filter_item in filters:
                        if filter_item['filterType'] == 'MIN_NOTIONAL':
                            min_notional = float(filter_item.get('notional', 10.0))
                
                # 현재 가격으로 주문 금액 확인
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                order_value = final_quantity * current_price
                
                if order_value < min_notional:
                    self.logger.warning(f"⚠️ {symbol} 주문 금액 부족, 조정: ${order_value:.2f} < ${min_notional}")
                    # 최소 주문 금액 맞추기
                    adjusted_quantity = (min_notional * 1.1) / current_price
                    final_quantity = round(adjusted_quantity, 6)
                    self.logger.info(f"📦 {symbol} 주문 수량 조정: {final_quantity:.6f}주")
                
                # 주문 실행
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=final_quantity
                )
                
                self.logger.info(f"✅ 주문 성공: {order.get('orderId', 'N/A')}")
                return {
                    'success': True,
                    'order_id': order.get('orderId'),
                    'executed_qty': float(order.get('executedQty', final_quantity)),
                    'avg_price': float(order.get('avgPrice', current_price)),
                    'status': order.get('status')
                }
                    
            except BinanceAPIException as e:
                error_code = e.code
                error_msg = e.message
                
                # 🔥 API 키 오류는 재시도 의미 없음
                if error_code == -2015:
                    self.logger.error(f"❌ API 키 오류로 주문 실패: {error_msg}")
                    return {
                        'success': False,
                        'error': f"API_KEY_ERROR: {error_msg}",
                        'retryable': False
                    }
                
                # 🔥 로트 사이즈 오류 (-1111)
                elif error_code == -1111:
                    try:
                        symbol_info = self.get_symbol_info(symbol)
                        step_size = 0.001
                        if symbol_info:
                            filters = symbol_info.get('filters', [])
                            for filter_item in filters:
                                if filter_item['filterType'] == 'LOT_SIZE':
                                    step_size = float(filter_item['stepSize'])
                                    break
                        
                        # 스텝 사이즈에 맞게 정확히 재조정
                        if step_size > 0:
                            adjusted_quantity = (quantity // step_size) * step_size
                            adjusted_quantity = round(adjusted_quantity, 6)
                            if adjusted_quantity <= 0:
                                adjusted_quantity = step_size
                        else:
                            adjusted_quantity = round(quantity * 0.95, 6)
                        
                        self.logger.warning(f"⚠️ 로트 사이즈 오류, 정확한 수량 조정: {quantity:.6f} -> {adjusted_quantity:.6f}")
                        quantity = adjusted_quantity
                        
                        # 마지막 시도가 아니라면 계속
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        else:
                            return {
                                'success': False,
                                'error': f"LOT_SIZE_ERROR: {error_msg}",
                                'retryable': False
                            }
                            
                    except Exception as adjust_error:
                        self.logger.error(f"❌ 수량 조정 실패: {adjust_error}")
                        quantity = round(quantity * 0.95, 6)
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        else:
                            return {
                                'success': False,
                                'error': f"ADJUSTMENT_FAILED: {error_msg}",
                                'retryable': False
                            }
                        
                # 🔥 잔고 부족 오류 (-2010)
                elif error_code == -2010:
                    self.logger.error(f"❌ 잔고 부족: {error_msg}")
                    return {
                        'success': False, 
                        'error': f"INSUFFICIENT_BALANCE: {error_msg}",
                        'retryable': False
                    }
                    
                # 🔥 그 외 오류는 재시도
                else:
                    self.logger.warning(f"⚠️ 주문 실패 ({error_code}): {error_msg}, 재시도 중...")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        return {
                            'success': False,
                            'error': f"API_ERROR_{error_code}: {error_msg}",
                            'retryable': True
                        }
                        
            except Exception as e:
                self.logger.error(f"❌ 주문 예외: {e}, 재시도 중...")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    return {
                        'success': False,
                        'error': f"EXCEPTION: {str(e)}",
                        'retryable': True
                    }
        
        return {
            'success': False,
            'error': 'MAX_RETRIES_EXCEEDED',
            'retryable': False
        }

    def safe_market_order(self, symbol: str, side: str, quantity: float, timeout: int = 10) -> Dict:
        """안전한 시장가 주문 실행"""
        result = [None]
        exception = [None]
        
        def execute_order():
            try:
                self.set_leverage(symbol, 10)
                
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=round(quantity, 6)
                )
                result[0] = order
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=execute_order)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            self.logger.error(f"⏰ {symbol} 주문 타임아웃 ({timeout}초)")
            return {'error': 'timeout'}
        
        if exception[0]:
            self.logger.error(f"❌ {symbol} 주문 실패: {exception[0]}")
            return {'error': str(exception[0])}
        
        return result[0] if result[0] else {'error': 'unknown'}


class MultiExchangeManager:
    """🔥 멀티 익스체인지 관리 클래스 - 실전 매매 최적화"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.exchanges = {}
        self.active_exchange = None
        self._cached_balances = {}
        self._initialize_exchanges()
                
    def _initialize_exchanges(self):
        """거래소 초기화 - 실전 매매 검증 강화"""
        try:
            load_dotenv()
            
            binance_enabled = self.config.get('binance', {}).get('trade_enabled', False)
            bybit_enabled = self.config.get('bybit', {}).get('trade_enabled', False)
            
            print(f"🔧 거래소 초기화 디버그: Binance={binance_enabled}, Bybit={bybit_enabled}")
            
            exchanges_initialized = []
            
            # Binance 초기화
            try:
                binance_key = os.getenv('BINANCE_API_KEY')
                binance_secret = "**********"
                
                print(f"🔧 Binance API 키 존재: {bool(binance_key)}")
                print(f"🔧 Binance 시크릿 존재: "**********"
                
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"b "**********"i "**********"n "**********"a "**********"n "**********"c "**********"e "**********"_ "**********"k "**********"e "**********"y "**********"  "**********"a "**********"n "**********"d "**********"  "**********"b "**********"i "**********"n "**********"a "**********"n "**********"c "**********"e "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********": "**********"
                    if binance_key in ['your_binance_api_key_here', 'test_binance_key']:
                        print("⚠️ Binance 테스트 API 키 감지")
                    else:
                        binance_exchange = BinanceExchange(
                            api_key=binance_key,
                            api_secret= "**********"
                            testnet=self.config.get('binance', {}).get('testnet', False)
                        )
                        
                        # Binance 연결 테스트
                        try:
                            test_balance = binance_exchange.get_futures_balance()
                            self.exchanges['binance'] = binance_exchange
                            self._cached_balances['binance'] = test_balance
                            exchanges_initialized.append('binance')
                            print(f"✅ Binance 거래소 초기화 완료 (잔고: ${test_balance:.2f})")
                        except Exception as test_e:
                            print(f"❌ Binance 연결 테스트 실패: {test_e}")
                            raise Exception(f"Binance 연결 실패: {test_e}")
                else:
                    print("❌ Binance API 키 또는 시크릿이 없습니다")
                    raise Exception("Binance API 키가 설정되지 않았습니다")
                    
            except Exception as e:
                print(f"❌ Binance 초기화 실패: {e}")
                raise
        
            if bybit_enabled:
                print("🚫 Bybit는 현재 비활성화 상태입니다")
        
            if exchanges_initialized:
                primary = self.config.get('exchange_config', {}).get('primary_exchange', 'binance')
                if primary in exchanges_initialized:
                    self.active_exchange = self.exchanges[primary]
                    print(f"🎯 기본 거래소 설정: {primary}")
                else:
                    self.active_exchange = self.exchanges[exchanges_initialized[0]]
                    print(f"🎯 자동 기본 거래소 설정: {exchanges_initialized[0]}")
                
                self._log_initial_balances()
            else:
                print("❌ 사용 가능한 거래소가 없습니다")
                raise Exception("사용 가능한 거래소가 없습니다")
                    
        except Exception as e:
            print(f"❌ 거래소 초기화 실패: {e}")
            self.active_exchange = None
            raise

    def _log_initial_balances(self):
        """초기 잔고 로깅"""
        total_balance = sum(self._cached_balances.values())
        balance_info = ", ".join([f"{exch}: ${bal:.2f}" for exch, bal in self._cached_balances.items()])
        self.logger.info(f"💰 초기 잔고 - 총계: ${total_balance:.2f} [{balance_info}]")
    
    def get_active_exchange(self):
        """현재 활성 거래소 반환"""
        return self.active_exchange
    
    def get_balance_all(self) -> Dict[str, float]:
        """모든 거래소 잔고 조회"""
        return self._cached_balances.copy()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("✅ trading_executor_v2.py 실전 매매 최적화 완료")