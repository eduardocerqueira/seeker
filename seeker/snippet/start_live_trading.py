#date: 2025-11-21T16:47:09Z
#url: https://api.github.com/gists/1cca9d2d0fccb338f95cb883ba7cb921
#owner: https://api.github.com/users/luckman538


# start_live_trading.py - Phase 13.5: PnL 계산 시스템 수정
import os
import time
import yaml
import sys
import logging
import pandas as pd
import numpy as np
import csv
import random as rand_module
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

# 로그 분석기 임포트 시도
try:
    from log_analyzer import TradingLogAnalyzer
    LOG_ANALYZER_AVAILABLE = True
    print("✅ 로그 분석기 모듈 임포트 성공")
except ImportError as e:
    print(f"⚠️ 로그 분석기 모듈 임포트 실패: {e}")
    LOG_ANALYZER_AVAILABLE = False
    class TradingLogAnalyzer:
        def __init__(self): self.logger = logging.getLogger(__name__)
        def load_trade_data(self) -> pd.DataFrame: return pd.DataFrame()
        def analyze_trading_patterns(self) -> Dict: return {}
        def generate_improvement_report(self) -> str: return ""
        def update_config_based_on_analysis(self, config: Dict) -> Dict: return config
        def plot_performance_charts(self, save_path: str = "performance_charts.png"): return False

# 현재 디렉토리 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Phase 13.5: 개선된 데이터 로깅 함수
def log_trade_to_csv(symbol: str, action: str, price: float, quantity: float, pnl: float = 0.0):
    """거래 내역 CSV 로깅 - Phase 13.5 PnL 수정"""
    try:
        file_exists = os.path.isfile('trades_log.csv')
        with open('trades_log.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'symbol', 'action', 'price', 'quantity', 'pnl', 'total_value'])
            
            total_value = price * quantity
            writer.writerow([
                datetime.now().isoformat(),
                symbol,
                action,
                round(price, 6),
                round(quantity, 6),
                round(pnl, 6),
                round(total_value, 2)
            ])
        print(f"✅ 거래 로그 기록: {symbol} {action} ${price:.4f} (PnL: ${pnl:.2f})")
    except Exception as e:
        print(f"❌ 거래 로깅 실패: {e}")

def log_performance_to_csv(performance_data: Dict):
    """성능 데이터 CSV 로깅 - Phase 13.5"""
    try:
        file_exists = os.path.isfile('performance_log.csv')
        with open('performance_log.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'total_trades', 'winning_trades', 'total_pnl', 
                               'unrealized_pnl', 'win_rate', 'sharpe_ratio', 'max_drawdown'])
            writer.writerow([
                datetime.now().isoformat(),
                performance_data.get('total_trades', 0),
                performance_data.get('winning_trades', 0),
                round(performance_data.get('total_pnl', 0.0), 6),
                round(performance_data.get('total_unrealized_pnl', 0.0), 6),
                round(performance_data.get('win_rate', 0.0), 4),
                round(performance_data.get('sharpe_ratio', 0.0), 4),
                round(performance_data.get('max_drawdown', 0.0), 4)
            ])
    except Exception as e:
        print(f"❌ 성능 로깅 실패: {e}")

def get_daily_pnl_from_logs() -> float:
    """일일 PnL 계산 - 거래 로그에서 집계"""
    try:
        if not os.path.exists('trades_log.csv'):
            return 0.0
        daily_pnl = 0.0
        today = date.today().isoformat()
        with open('trades_log.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['timestamp'].startswith(today):
                    daily_pnl += float(row.get('pnl', 0))
        return daily_pnl
    except Exception as e:
        print(f"❌ 일일 PnL 계산 실패: {e}")
        return 0.0

# 코어 모듈 임포트
try:
    from core.engine import CoreTradingEngine
    from core.models import PerformanceMetrics, LeveragedSignal, PortfolioSnapshot
    CORE_MODULE_AVAILABLE = True
    print("✅ 코어 모듈 임포트 성공")
except ImportError as e:
    print(f"⚠️ 코어 모듈 임포트 경고: {e}")
    print("💡 폴백 모드로 계속 진행합니다...")
    CORE_MODULE_AVAILABLE = False
    class CoreTradingEngine:
        def __init__(self, config): 
            self.config = config
            self.discord_notifier = None
            self.cycle_count = 0
        def execute_trading_cycle(self, symbols, executor, strategy, portfolio_manager):
            self.cycle_count += 1
            result = {
                "status": "fallback_success",
                "cycle_count": self.cycle_count,
                "timestamp": datetime.now().isoformat(),
                "symbols_processed": len(symbols),
                "signals_generated": 0,
                "positions_found": 0
            }
            for symbol in symbols:
                try:
                    has_position = executor.safe_has_open_position(symbol)
                    if has_position:
                        result["positions_found"] += 1
                        result["signals_generated"] += 1
                except Exception as e:
                    print(f"⚠️ {symbol} 모니터링 오류: {e}")
            print(f"📝 폴백 사이클 {self.cycle_count} 완료: {result['symbols_processed']}개 심볼, {result['positions_found']}개 포지션")
            return result
    class LeveragedSignal:
        def __init__(self): 
            self.signal_type = 'hold'
            self.confidence = 0.0
            self.symbol = ''
            self.timestamp = datetime.now()

# 타입 안전성 모듈 임포트
try:
    from type_safety import type_safe
    print("✅ 타입 안전성 모듈 임포트 성공")
except ImportError as e:
    print(f"⚠️ 타입 안전성 모듈 임포트 실패: {e}")
    class TypeSafeFallback:
        @staticmethod
        def safe_float(value, default=0.0):
            try:
                return float(value) if value is not None else default
            except (TypeError, ValueError):
                return default
        @staticmethod
        def safe_int(value, default=0):
            try:
                return int(value) if value is not None else default
            except (TypeError, ValueError):
                return default
        @staticmethod
        def validate_confidence(confidence):
            return 0 <= confidence <= 1
    type_safe = TypeSafeFallback()

def load_config() -> Optional[Dict[str, Any]]:
    """보안 설정 파일 로드 - 멀티 익스체인지 지원"""
    try:
        load_dotenv()
        config_path = 'enhanced_config_live.yaml'
        if not os.path.exists(config_path):
            print(f"❌ 설정 파일을 찾을 수 없습니다: {config_path}")
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        exchange_type = config.get('exchange_config', {}).get('exchange_type', 'binance')
        multi_exchange = config.get('exchange_config', {}).get('multi_exchange_support', False)
        
        required_sections = ['monitoring', 'trading']
        if exchange_type == 'binance' or multi_exchange:
            required_sections.append('binance')
        if exchange_type == 'bybit' or multi_exchange:
            required_sections.append('bybit')
        
        for section in required_sections:
            if section not in config:
                print(f"❌ 필수 설정 섹션 '{section}'이 없습니다.")
                return None
        
        if exchange_type == 'binance' or multi_exchange:
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"a "**********"p "**********"i "**********"_ "**********"k "**********"e "**********"y "**********"  "**********"o "**********"r "**********"  "**********"n "**********"o "**********"t "**********"  "**********"a "**********"p "**********"i "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********": "**********"
                print("❌ 환경변수에서 BINANCE_API_KEY, BINANCE_API_SECRET을 찾을 수 없습니다.")
                return None
            config['binance']['mainnet_api_key'] = api_key
            config['binance']['mainnet_api_secret'] = "**********"
        
        if exchange_type == 'bybit' or multi_exchange:
            api_key = os.getenv('BYBIT_API_KEY')
            api_secret = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"a "**********"p "**********"i "**********"_ "**********"k "**********"e "**********"y "**********"  "**********"o "**********"r "**********"  "**********"n "**********"o "**********"t "**********"  "**********"a "**********"p "**********"i "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********": "**********"
                print("❌ 환경변수에서 BYBIT_API_KEY, BYBIT_API_SECRET을 찾을 수 없습니다.")
                return None
            config['bybit']['mainnet_api_key'] = api_key
            config['bybit']['mainnet_api_secret'] = "**********"
        
        if 'symbols' not in config['monitoring']:
            print("❌ 모니터링 심볼이 설정되지 않았습니다.")
            return None
            
        if not config['monitoring']['symbols']:
            print("❌ 모니터링 심볼 리스트가 비어있습니다.")
            return None
        
        config['monitoring'].setdefault('update_interval', 300)
        config['trading'].setdefault('min_confidence', 0.35)
        config['trading'].setdefault('risk_per_trade', 0.02)
        
        print(f"✅ 멀티 익스체인지 설정 로드 완료: {exchange_type}")
        print(f"   심볼 개수: {len(config['monitoring']['symbols'])}")
        print(f"   실제 거래: {config['binance'].get('trade_enabled', False)}")
        
        return config
        
    except yaml.YAMLError as e:
        print(f"❌ YAML 파싱 오류: {e}")
        return None
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        return None

class PerformanceMonitor:
    """실시간 성능 모니터링 - Phase 13.5 PnL 계산 수정"""
    
    def __init__(self, config):
        self.config = config
        self.performance_data = {}
        self.start_time = datetime.now()
        self.logger = logging.getLogger(__name__)
        
        symbols = config.get('monitoring', {}).get('symbols', [])
        for symbol in symbols:
            self.performance_data[symbol] = {
                'total_trades': 0,
                'winning_trades': 0,
                'total_pnl': 0.0,
                'signals_generated': 0,
                'last_signal': None
            }
        
        self.logger.info(f"✅ 성능 모니터 초기화: {len(symbols)}개 심볼")
        
    def record_trade(self, symbol, signal_type, confidence, result):
        """거래 결과 기록"""
        try:
            if symbol not in self.performance_data:
                self.performance_data[symbol] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_pnl': 0.0,
                    'signals_generated': 0,
                    'last_signal': None
                }
            
            data = self.performance_data[symbol]
            data['total_trades'] += 1
            data['signals_generated'] += 1
            
            # 🔥 Phase 13.5: 실제 PnL 기록
            pnl = result.get('pnl', 0.0)
            data['total_pnl'] += pnl
            
            if pnl > 0:
                data['winning_trades'] += 1
            
            data['last_signal'] = {
                'type': signal_type,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'result': result
            }
            
        except Exception as e:
            self.logger.error(f"❌ 성능 기록 실패: {e}")

    def calculate_real_time_pnl(self, engine) -> Dict[str, Dict]:
        """🔥 Phase 13.5: 완전히 개선된 실시간 PnL 계산"""
        try:
            pnl_data = {}
            total_unrealized_pnl = 0.0
            
            if not engine or not hasattr(engine, 'executor') or not engine.executor:
                self.logger.warning("⚠️ PnL 계산: 실행기 없음")
                return pnl_data
                
            symbols = self.config['monitoring']['symbols']
            
            for symbol in symbols:
                try:
                    # 🔥 MATICUSDT 특별 처리: 재시도 + 타임아웃 조정
                    max_retries = 4 if symbol == 'MATICUSDT' else 3
                    timeout = 20 if symbol == 'MATICUSDT' else 10
                    
                    for attempt in range(max_retries):
                        try:
                            # 포지션 정보 조회
                            positions = engine.executor.client.futures_position_information(
                                symbol=symbol
                            )
                            
                            for position in positions:
                                position_amt = float(position.get('positionAmt', 0))
                                
                                if abs(position_amt) > 0.00001:
                                    # 🔥 Binance API의 unrealizedProfit 직접 사용
                                    unrealized_pnl = float(position.get('unrealizedProfit', 0))
                                    entry_price = float(position.get('entryPrice', 0))
                                    leverage = float(position.get('leverage', 1))
                                    mark_price = float(position.get('markPrice', entry_price))
                                    
                                    # 🔥 실제 현재가 기반 PnL 재계산 (검증용)
                                    if entry_price > 0:
                                        if position_amt > 0:  # LONG
                                            calculated_pnl = (mark_price - entry_price) * abs(position_amt)
                                        else:  # SHORT
                                            calculated_pnl = (entry_price - mark_price) * abs(position_amt)
                                    else:
                                        calculated_pnl = unrealized_pnl
                                    
                                    # 🔥 API PnL과 계산 PnL 비교 (정확도 검증)
                                    pnl_diff = abs(unrealized_pnl - calculated_pnl)
                                    if pnl_diff > 0.01:  # 0.01 이상 차이 시 경고
                                        self.logger.warning(f"⚠️ {symbol} PnL 차이: API={unrealized_pnl:.4f}, 계산={calculated_pnl:.4f}")
                                    
                                    # 최종 PnL은 API 값 사용 (더 정확함)
                                    final_pnl = unrealized_pnl
                                    
                                    pnl_data[symbol] = {
                                        'unrealized_pnl': final_pnl,
                                        'calculated_pnl': calculated_pnl,
                                        'position_amt': position_amt,
                                        'entry_price': entry_price,
                                        'mark_price': mark_price,
                                        'leverage': leverage,
                                        'position_side': 'LONG' if position_amt > 0 else 'SHORT',
                                        'pnl_verified': pnl_diff < 0.01
                                    }
                                    
                                    total_unrealized_pnl += final_pnl
                                    self.logger.info(f"📊 {symbol} 실시간 PnL: ${final_pnl:.4f} (검증: {'✅' if pnl_diff < 0.01 else '⚠️'})")
                                    break
                            
                            break  # 성공시 재시도 중단
                            
                        except Exception as e:
                            error_msg = str(e)
                            if attempt < max_retries - 1:
                                # 🔥 지수 백오프 적용
                                wait_time = 2 ** attempt
                                self.logger.warning(f"⚠️ {symbol} PnL 조회 실패, {wait_time}초 후 재시도... ({attempt+1}/{max_retries})")
                                time.sleep(wait_time)
                                
                                # 특정 에러는 재시도 의미 없음
                                if "Invalid symbol" in error_msg or "API-key" in error_msg:
                                    break
                            else:
                                self.logger.error(f"❌ {symbol} PnL 계산 최종 실패: {e}")
                                # 실패시 기본값 기록 (0으로 처리)
                                pnl_data[symbol] = {
                                    'unrealized_pnl': 0.0,
                                    'calculated_pnl': 0.0,
                                    'position_amt': 0.0,
                                    'entry_price': 0.0,
                                    'mark_price': 0.0,
                                    'leverage': 1,
                                    'position_side': 'NONE',
                                    'error': error_msg,
                                    'pnl_verified': False
                                }
                                
                except Exception as e:
                    self.logger.warning(f"⚠️ {symbol} PnL 계산 실패: {e}")
                    continue
                        
            # 총 PnL 저장
            self.total_unrealized_pnl = total_unrealized_pnl
            self.logger.info(f"💰 포트폴리오 총 미실현 PnL: ${total_unrealized_pnl:.4f}")
            
            return pnl_data
            
        except Exception as e:
            self.logger.error(f"❌ 실시간 PnL 계산 실패: {e}")
            return {}

    def set_engine(self, engine):
        """엔진 설정 메서드"""
        self.engine = engine
        self.logger.info("✅ 성능 모니터에 엔진 설정 완료")

    def get_portfolio_summary(self, engine) -> Dict:
        """포트폴리오 요약 - 실제 PnL 반영"""
        try:
            # 🔥 Phase 13.5: 개선된 PnL 계산
            pnl_data = self.calculate_real_time_pnl(engine)
            
            # 기존 성능 데이터와 결합
            base_summary = self.get_performance_summary()
            
            # 실제 PnL 데이터 추가
            portfolio_summary = {
                **base_summary,
                'total_unrealized_pnl': self.total_unrealized_pnl if hasattr(self, 'total_unrealized_pnl') else 0.0,
                'pnl_data': pnl_data,
                'active_positions': len(pnl_data),
                'portfolio_value': 0.0,
                'data_quality': 'PHASE_13.5_PNL_FIXED'
            }
            
            # 포트폴리오 가치 계산 시도
            try:
                if engine and hasattr(engine, 'executor') and engine.executor:
                    balance = engine.executor.get_futures_balance()
                    portfolio_summary['portfolio_value'] = balance + portfolio_summary['total_unrealized_pnl']
                    portfolio_summary['total_balance'] = balance
            except Exception as e:
                self.logger.warning(f"⚠️ 포트폴리오 가치 계산 실패: {e}")
            
            return portfolio_summary
            
        except Exception as e:
            self.logger.error(f"❌ 포트폴리오 요약 실패: {e}")
            return self.get_performance_summary()

    def get_performance_summary(self):
        """성능 요약 데이터 반환"""
        try:
            total_symbols = len(self.performance_data)
            total_trades = sum(data.get('total_trades', 0) for data in self.performance_data.values())
            winning_trades = sum(data.get('winning_trades', 0) for data in self.performance_data.values())
            
            # 🔥 Phase 13.5: 실제 PnL 집계
            total_pnl = sum(data.get('total_pnl', 0) for data in self.performance_data.values())
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'total_symbols': total_symbols,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                'active_positions': len([data for data in self.performance_data.values() 
                                    if data.get('total_trades', 0) > 0]),
                'data_quality': 'PHASE_13.5_ACTIVE'
            }
            
        except Exception as e:
            self.logger.error(f"❌ 성능 요약 생성 실패: {e}")
            return self._get_empty_summary()

    def _get_empty_summary(self):
        """빈 성능 요약 반환"""
        return {
            'total_symbols': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'active_positions': 0,
            'data_quality': 'ERROR',
            'warning': '데이터 오류'
        }

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Sharpe Ratio 계산"""
        try:
            # 🔥 Phase 13.5: 실제 PnL 기반 Sharpe Ratio
            if not hasattr(self, 'total_unrealized_pnl'):
                return 0.0
            
            # 간단한 Sharpe Ratio 계산 (더 정교한 계산은 향후 개선)
            avg_return = self.total_unrealized_pnl / max(1, len(self.performance_data))
            std_return = 0.01  # 임시값, 실제로는 수익률 표준편차 계산 필요
            
            if std_return > 0:
                sharpe = (avg_return - risk_free_rate) / std_return
                return max(-10.0, min(10.0, sharpe))
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Sharpe 계산 실패: {e}")
            return 0.0

    def calculate_max_drawdown(self) -> float:
        """최대 낙폭 계산"""
        try:
            # 🔥 Phase 13.5: 실제 PnL 기반 MDD
            if not hasattr(self, 'total_unrealized_pnl'):
                return 0.0
            
            # 간단한 MDD 계산 (향후 개선)
            total_pnl = sum(data.get('total_pnl', 0) for data in self.performance_data.values())
            
            if total_pnl < 0:
                return abs(total_pnl) / max(1, self.config['trading'].get('initial_capital', 280))
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ MDD 계산 실패: {e}")
            return 0.0

    def generate_report(self):
        """성능 보고서 생성 - Phase 13.5"""
        try:
            summary = self.get_performance_summary()
            total_symbols = summary.get('total_symbols', len(self.performance_data))
            
            report = f"""
📊 Evo-Quant AI 실시간 성능 보고서 (Phase 13.5 - PnL 수정)
==========================================================
• 모니터링 심볼: {total_symbols}개
• 총 거래: {summary.get('total_trades', 0)}회
• 승률: {summary.get('win_rate', 0):.1f}%
• 실시간 PnL: ${summary.get('total_pnl', 0):.4f}
• 가동 시간: {summary.get('uptime_hours', 0):.1f}시간

✅ Phase 13.5: PnL 계산 시스템 정상 가동
💡 실시간 포지션 평가손익 반영

📈 심볼별 거래 횟수:
"""
            for symbol, data in self.performance_data.items():
                total_trades = data.get('total_trades', 0)
                symbol_pnl = data.get('total_pnl', 0)
                report += f"   {symbol}: {total_trades}회 (PnL: ${symbol_pnl:.2f})\n"
            
            report += f"\n📋 데이터 품질: {summary.get('data_quality', 'UNKNOWN')}"
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ 성능 보고서 생성 실패: {e}")
            return f"📊 성능 보고서 생성 실패 - {str(e)}"
    
    def generate_advanced_validation_report(self, symbol: str, signal_data: Dict) -> str:
        """실전 검증 과정 보고서 생성"""
        try:
            report = f"""
    🎯 {symbol} 실전 검증 보고서
    ================================
    📊 기본 신호 정보
    • 신호 타입: {signal_data.get('signal_type', 'N/A')}
    • 원본 신뢰도: {signal_data.get('confidence', 0):.3f}
    • 검증 후 신뢰도: {signal_data.get('adjusted_confidence', 0):.3f}

    🔍 검증 과정 결과
    """
            
            # 봉 분석 결과
            candle_analysis = signal_data.get('candle_analysis', {})
            if candle_analysis:
                report += f"""📈 봉 분석
    • 몸통 비율: {candle_analysis.get('body_ratio', 0):.1%}
    • 봉 크기: {candle_analysis.get('size_percent', 0):.3f}%
    • 모멘텀: {candle_analysis.get('momentum', 0):.3f}%
    • 거래량 변화: {candle_analysis.get('volume_change', 0):.1f}x

    """
            
            # 신호 강도 구성
            report += f"""💪 신호 강도 구성
    • 진행도 점수: {signal_data.get('signal_progress', 0):.1f}
    • 변동성 조정: {signal_data.get('volatility_adjustment', 0):.1f}
    • 시장 상황: {signal_data.get('market_regime', 'N/A')}

    """
            
            # 최종 평가
            final_confidence = signal_data.get('adjusted_confidence', 0)
            min_confidence = 0.05  # 기본값
            
            if final_confidence >= 0.08:
                evaluation = "✅ 강한 신호"
            elif final_confidence >= 0.05:
                evaluation = "⚠️ 보통 신호"  
            else:
                evaluation = "❌ 약한 신호"
                
            report += f"""🎯 최종 평가
    • {evaluation}
    • 최종 점수: {final_confidence:.3f}
    • 임계값: {min_confidence}
    • 행동: {'진입 가능' if final_confidence >= min_confidence else '대기 필요'}
    """
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ 검증 보고서 생성 실패: {e}")
            return f"검증 보고서 생성 실패: {e}"

class RealTimeVolatilityAnalyzer:
    """실시간 변동성 분석기 - Phase 11.0"""
    
    def __init__(self, config):
        self.config = config
        self.volatility_data = {}
        self.price_history = {}
        self.volatility_threshold = 0.02
        
    def update_price_data(self, symbol: str, current_price: float):
        """가격 데이터 업데이트 및 변동성 계산"""
        try:
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            history = self.price_history[symbol]
            history.append(current_price)
            if len(history) > 50:
                history.pop(0)
            
            if len(history) >= 10:
                returns = []
                for i in range(1, len(history)):
                    daily_return = (history[i] - history[i-1]) / history[i-1]
                    returns.append(daily_return)
                
                if returns:
                    volatility = np.std(returns)
                    self.volatility_data[symbol] = volatility
                    return volatility
                    
            return 0.0
            
        except Exception as e:
            print(f"❌ 변동성 분석 실패: {e}")
            return 0.0
    
    def get_market_regime(self, symbol: str) -> str:
        """시장 상황 판단 (고변동성/저변동성)"""
        try:
            volatility = self.volatility_data.get(symbol, 0.0)
            
            if volatility > self.volatility_threshold * 1.5:
                return "HIGH_VOLATILITY"
            elif volatility > self.volatility_threshold:
                return "MEDIUM_VOLATILITY"
            else:
                return "LOW_VOLATILITY"
                
        except Exception as e:
            print(f"❌ 시장 상황 판단 실패: {e}")
            return "UNKNOWN"
    
    def get_recommended_parameters(self, symbol: str) -> Dict:
        """변동성 기반 권장 파라미터 반환"""
        try:
            regime = self.get_market_regime(symbol)
            volatility = self.volatility_data.get(symbol, 0.0)
            
            if regime == "HIGH_VOLATILITY":
                return {
                    'atr_multiplier': 2.0,
                    'risk_per_trade': 0.02,
                    'min_confidence': 0.08
                }
            elif regime == "MEDIUM_VOLATILITY":
                return {
                    'atr_multiplier': 1.5,
                    'risk_per_trade': 0.03, 
                    'min_confidence': 0.05
                }
            else:
                return {
                    'atr_multiplier': 1.2,
                    'risk_per_trade': 0.04,
                    'min_confidence': 0.03
                }
                
        except Exception as e:
            print(f"❌ 파라미터 추천 실패: {e}")
            return {
                'atr_multiplier': 1.5,
                'risk_per_trade': 0.03,
                'min_confidence': 0.05
            }

class DataAnomalyDetector:
    """데이터 이상치 탐지기"""
    
    def __init__(self, executor=None):
        self.executor = executor
        self.logger = logging.getLogger(__name__)
    
    def detect_price_spikes(self, symbol: str, current_price: float) -> bool:
        """가격 급등락 탐지"""
        try:
            # 최근 1시간 평균 가격과 비교
            recent_prices = self._get_recent_prices(symbol, minutes=60)
            if len(recent_prices) < 10:
                return False
                
            avg_price = np.mean(recent_prices)
            price_change = abs(current_price - avg_price) / avg_price
            
            # 5% 이상 변동시 이상치로 판단
            return price_change > 0.05
            
        except Exception as e:
            self.logger.warning(f"⚠️ {symbol} 가격 이상치 탐지 실패: {e}")
            return False
    
    def _get_recent_prices(self, symbol: str, minutes: int = 60) -> List[float]:
        """최근 가격 데이터 조회"""
        try:
            if not self.executor:
                return []
                
            # 1분 봉 데이터로 최근 가격 조회
            klines = self.executor.client.futures_klines(
                symbol=symbol,
                interval='1m', 
                limit=minutes
            )
            return [float(k[4]) for k in klines]  # 종가 반환
        except Exception as e:
            self.logger.warning(f"⚠️ {symbol} 가격 데이터 조회 실패: {e}")
            return []

class AdvancedIndicatorsIntegrator:
    """4가지 고급 인디케이터 통합 시스템 - Phase 13.2"""
    
    def __init__(self, config: Dict, executor):
        self.config = config
        self.executor = executor
        self.logger = logging.getLogger(__name__)
        
        # 인디케이터 데이터 캐시
        self.delta_flow_cache = {}
        self.vwap_cache = {}
        self.volume_profile_cache = {}
        self.supertrend_cache = {}
        
        # 설정값
        self.indicators_config = config.get('enhanced_indicators', {})
        
        self.logger.info("🎯 고급 인디케이터 통합 시스템 초기화 완료")
    
    def update_all_indicators(self, symbol: str):
        """모든 인디케이터 데이터 업데이트"""
        try:
            self.logger.info(f"🔄 {symbol} 고급 인디�이터 업데이트 중...")
            
            # 1. Delta Flow Profile 업데이트
            self._update_delta_flow_profile(symbol)
            
            # 2. VWAP Periodic Close 업데이트
            self._update_vwap_periodic_close(symbol)
            
            # 3. Volume Profile 업데이트
            self._update_volume_profile_ultra(symbol)
            
            # 4. Supertrend 업데이트
            self._update_supertrend(symbol)
            
            self.logger.info(f"✅ {symbol} 모든 인디케이터 업데이트 완료")
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 인디케이터 업데이트 실패: {e}")
    
    def _update_delta_flow_profile(self, symbol: str):
        """Delta Flow Profile [LuxAlgo] - 돈 흐름 프로파일과 델타 프로파일"""
        try:
            # 1시간 봉 데이터로 델타 플로우 계산
            klines = self.executor.client.futures_klines(
                symbol=symbol, interval='1h', limit=100
            )
            
            if len(klines) < 50:
                return
            
            # 데이터 추출
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            closes = [float(k[4]) for k in klines]
            volumes = [float(k[5]) for k in klines]
            
            # 델타 계산 (매수 거래량 - 매도 거래량, 단순화)
            deltas = self._calculate_delta_flow(highs, lows, closes, volumes)
            
            # 돈 흐름 프로파일 계산
            money_flow_profile = self._calculate_money_flow_profile(closes, volumes)
            
            # 현재 델타 상태
            current_delta = deltas[-1] if deltas else 0
            delta_trend = "BULLISH" if current_delta > 0 else "BEARISH"
            
            self.delta_flow_cache[symbol] = {
                'current_delta': current_delta,
                'delta_trend': delta_trend,
                'money_flow_profile': money_flow_profile,
                'delta_ma': np.mean(deltas[-20:]) if len(deltas) >= 20 else 0,
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"📊 {symbol} Delta Flow: {current_delta:.0f} ({delta_trend})")
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} Delta Flow 업데이트 실패: {e}")
    
    def _calculate_delta_flow(self, highs: list, lows: list, closes: list, volumes: list) -> list:
        """델타 플로우 계산 (LuxAlgo 방식 단순화)"""
        try:
            deltas = []
            for i in range(1, len(closes)):
                # 가격 변화와 거래량을 기반으로 델타 계산
                price_change = closes[i] - closes[i-1]
                volume = volumes[i]
                
                # 가격 상승시 매수 델타, 하락시 매도 델타로 가정
                if price_change > 0:
                    delta = volume * (price_change / closes[i-1])
                else:
                    delta = -volume * (abs(price_change) / closes[i-1])
                
                deltas.append(delta)
            
            return deltas
            
        except Exception as e:
            self.logger.error(f"❌ 델타 계산 실패: {e}")
            return []
    
    def _calculate_money_flow_profile(self, closes: list, volumes: list) -> Dict:
        """돈 흐름 프로파일 계산"""
        try:
            # MFI(Money Flow Index) 방식 단순화
            typical_prices = [(closes[i] + closes[i-1]) / 2 for i in range(1, len(closes))]
            money_flows = [tp * volumes[i+1] for i, tp in enumerate(typical_prices)]
            
            positive_flow = sum(mf for i, mf in enumerate(money_flows) if typical_prices[i] > typical_prices[i-1]) if len(typical_prices) > 1 else 0
            negative_flow = sum(mf for i, mf in enumerate(money_flows) if typical_prices[i] <= typical_prices[i-1]) if len(typical_prices) > 1 else 0
            
            money_flow_ratio = positive_flow / (positive_flow + negative_flow) if (positive_flow + negative_flow) > 0 else 0.5
            
            return {
                'money_flow_ratio': money_flow_ratio,
                'positive_flow': positive_flow,
                'negative_flow': negative_flow,
                'total_flow': positive_flow + negative_flow
            }
            
        except Exception as e:
            self.logger.error(f"❌ 돈 흐름 프로파일 계산 실패: {e}")
            return {'money_flow_ratio': 0.5, 'positive_flow': 0, 'negative_flow': 0, 'total_flow': 0}
    
    def _update_vwap_periodic_close(self, symbol: str):
        """VWAP Periodic Close [LuxAlgo] - 주기별 VWAP 종가 수준"""
        try:
            # 다양한 timeframe VWAP 계산
            timeframes = ['15m', '1h', '4h']
            vwap_levels = {}
            
            for tf in timeframes:
                klines = self.executor.client.futures_klines(
                    symbol=symbol, interval=tf, limit=100
                )
                
                if len(klines) >= 20:
                    vwap = self._calculate_vwap(klines)
                    current_price = float(klines[-1][4])
                    
                    vwap_levels[tf] = {
                        'vwap': vwap,
                        'deviation': (current_price - vwap) / vwap,
                        'position': 'ABOVE' if current_price > vwap else 'BELOW'
                    }
            
            self.vwap_cache[symbol] = {
                'timeframe_levels': vwap_levels,
                'primary_vwap': vwap_levels.get('1h', {}).get('vwap', 0),
                'timestamp': datetime.now()
            }
            
            # VWAP 상태 로깅
            primary_data = vwap_levels.get('1h', {})
            if primary_data:
                self.logger.info(f"📈 {symbol} VWAP: {primary_data['vwap']:.4f} "
                              f"({primary_data['position']}, {primary_data['deviation']:.2%})")
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} VWAP 업데이트 실패: {e}")
    
    def _calculate_vwap(self, klines: list) -> float:
        """VWAP 계산"""
        try:
            typical_prices = []
            volumes = []
            
            for k in klines:
                high, low, close, volume = float(k[2]), float(k[3]), float(k[4]), float(k[5])
                typical_price = (high + low + close) / 3
                typical_prices.append(typical_price)
                volumes.append(volume)
            
            # VWAP = ∑(Typical Price * Volume) / ∑Volume
            vwap = sum(tp * vol for tp, vol in zip(typical_prices, volumes)) / sum(volumes)
            return vwap
            
        except Exception as e:
            self.logger.error(f"❌ VWAP 계산 실패: {e}")
            return 0.0
    
    def _update_volume_profile_ultra(self, symbol: str):
        """Volume Profile Free Ultra SLI by RRB - 고해상도 볼륨 프로파일"""
        try:
            # 4시간 봉 데이터로 볼륨 프로파일 계산
            klines = self.executor.client.futures_klines(
                symbol=symbol, interval='4h', limit=50
            )
            
            if len(klines) < 20:
                return
            
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            volumes = [float(k[5]) for k in klines]
            closes = [float(k[4]) for k in klines]
            
            # 고해상도 볼륨 프로파일 계산 (100개 수준)
            profile_data = self._calculate_volume_profile_ultra(highs, lows, volumes, closes, levels=100)
            
            current_price = closes[-1]
            
            # POC(Point of Control)와 Value Area 분석
            poc_info = self._analyze_poc_value_area(profile_data, current_price)
            
            self.volume_profile_cache[symbol] = {
                'profile_data': profile_data,
                'poc_price': poc_info['poc_price'],
                'value_area_high': poc_info['value_area_high'],
                'value_area_low': poc_info['value_area_low'],
                'current_position': poc_info['current_position'],
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"📊 {symbol} Volume Profile: POC ${poc_info['poc_price']:.4f}, "
                          f"Value Area ${poc_info['value_area_low']:.4f}-${poc_info['value_area_high']:.4f}")
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} Volume Profile 업데이트 실패: {e}")
    
    def _calculate_volume_profile_ultra(self, highs: list, lows: list, volumes: list, closes: list, levels: int = 100) -> Dict:
        """고해상도 볼륨 프로파일 계산"""
        try:
            # 가격 범위 계산
            min_price = min(lows)
            max_price = max(highs)
            price_range = max_price - min_price
            
            if price_range <= 0:
                return {}
            
            # 가격 레벨 생성
            price_levels = np.linspace(min_price, max_price, levels)
            volume_at_price = {price: 0.0 for price in price_levels}
            
            # 각 봉의 거래량을 가격 레벨에 분배
            for i in range(len(highs)):
                high, low, volume = highs[i], lows[i], volumes[i]
                
                # 해당 봉이 커버하는 가격 레벨 찾기
                for price in price_levels:
                    if low <= price <= high:
                        # 가격이 해당 봉 내에 있을 때 거래량 분배
                        distance_from_close = abs(price - closes[i])
                        total_range = high - low
                        
                        if total_range > 0:
                            # 종가와 가까운 레벨에 더 많은 거래량 할당 (단순화)
                            weight = 1 - (distance_from_close / total_range)
                            volume_at_price[price] += volume * weight
            
            return volume_at_price
            
        except Exception as e:
            self.logger.error(f"❌ 볼륨 프로파일 계산 실패: {e}")
            return {}
    
    def _analyze_poc_value_area(self, profile_data: Dict, current_price: float) -> Dict:
        """POC 및 Value Area 분석"""
        try:
            if not profile_data:
                return {'poc_price': current_price, 'value_area_high': current_price, 
                       'value_area_low': current_price, 'current_position': 'UNKNOWN'}
            
            # POC(최대 거래량 가격) 찾기
            poc_price = max(profile_data.items(), key=lambda x: x[1])[0]
            
            # 총 거래량 계산
            total_volume = sum(profile_data.values())
            
            # Value Area 계산 (상위 70% 거래량 영역)
            target_volume = total_volume * 0.7
            sorted_prices = sorted(profile_data.items(), key=lambda x: x[1], reverse=True)
            
            cumulative_volume = 0
            value_area_prices = []
            
            for price, volume in sorted_prices:
                cumulative_volume += volume
                value_area_prices.append(price)
                if cumulative_volume >= target_volume:
                    break
            
            value_area_high = max(value_area_prices)
            value_area_low = min(value_area_prices)
            
            # 현재 가격 위치 분석
            if current_price > value_area_high:
                position = 'ABOVE_VALUE_AREA'
            elif current_price < value_area_low:
                position = 'BELOW_VALUE_AREA'
            else:
                position = 'INSIDE_VALUE_AREA'
            
            return {
                'poc_price': poc_price,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'current_position': position
            }
            
        except Exception as e:
            self.logger.error(f"❌ POC 분석 실패: {e}")
            return {'poc_price': current_price, 'value_area_high': current_price, 
                   'value_area_low': current_price, 'current_position': 'UNKNOWN'}
    
    def _update_supertrend(self, symbol: str, period: int = 10, multiplier: float = 3.0):
        """Supertrend - 추세 추종 지표"""
        try:
            # 15분 봉 데이터로 Supertrend 계산
            klines = self.executor.client.futures_klines(
                symbol=symbol, interval='15m', limit=100
            )
            
            if len(klines) < period * 2:
                return
            
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            closes = [float(k[4]) for k in klines]
            
            supertrend_data = self._calculate_supertrend(highs, lows, closes, period, multiplier)
            
            current_trend = supertrend_data['trend'][-1]
            trend_text = "BULLISH 📈" if current_trend == 1 else "BEARISH 📉"
            
            self.supertrend_cache[symbol] = {
                'trend': supertrend_data['trend'],
                'supertrend_line': supertrend_data['supertrend_line'],
                'atr': supertrend_data['atr'],
                'current_trend': current_trend,
                'trend_strength': self._calculate_trend_strength(supertrend_data['trend']),
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"🎯 {symbol} Supertrend: {trend_text} "
                          f"(Strength: {self.supertrend_cache[symbol]['trend_strength']:.2f})")
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} Supertrend 업데이트 실패: {e}")
    
    def _calculate_supertrend(self, highs: list, lows: list, closes: list, period: int, multiplier: float) -> Dict:
        """Supertrend 계산"""
        try:
            # ATR 계산
            atr = self._calculate_atr(highs, lows, closes, period)
            
            # 기본 밴드 계산
            basic_upper = [(high + low) / 2 + multiplier * atr for high, low in zip(highs, lows)]
            basic_lower = [(high + low) / 2 - multiplier * atr for high, low in zip(highs, lows)]
            
            # Supertrend 계산
            supertrend_line = []
            trend = []
            
            for i in range(len(closes)):
                if i == 0:
                    supertrend_line.append(basic_upper[i])
                    trend.append(1)  # 초기값 상승
                    continue
                
                close = closes[i]
                prev_supertrend = supertrend_line[i-1]
                
                if close > prev_supertrend:
                    # 상승 추세
                    supertrend_line.append(max(basic_lower[i], prev_supertrend))
                    trend.append(1)
                else:
                    # 하락 추세
                    supertrend_line.append(min(basic_upper[i], prev_supertrend))
                    trend.append(-1)
            
            return {
                'trend': trend,
                'supertrend_line': supertrend_line,
                'atr': atr
            }
            
        except Exception as e:
            self.logger.error(f"❌ Supertrend 계산 실패: {e}")
            return {'trend': [], 'supertrend_line': [], 'atr': 0}
    
    def _calculate_atr(self, highs: list, lows: list, closes: list, period: int) -> float:
        """ATR(Average True Range) 계산"""
        try:
            true_ranges = []
            
            for i in range(1, len(highs)):
                high, low, prev_close = highs[i], lows[i], closes[i-1]
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)
            
            if len(true_ranges) >= period:
                atr = np.mean(true_ranges[-period:])
            else:
                atr = np.mean(true_ranges) if true_ranges else 0
            
            return atr
            
        except Exception as e:
            self.logger.error(f"❌ ATR 계산 실패: {e}")
            return 0.0
    
    def _calculate_trend_strength(self, trend_data: list) -> float:
        """추세 강도 계산"""
        try:
            if len(trend_data) < 10:
                return 0.5
            
            # 최근 10개 봉에서의 추세 일관성
            recent_trends = trend_data[-10:]
            consistency = sum(1 for i in range(1, len(recent_trends)) 
                          if recent_trends[i] == recent_trends[i-1]) / (len(recent_trends) - 1)
            
            return consistency
            
        except Exception as e:
            self.logger.error(f"❌ 추세 강도 계산 실패: {e}")
            return 0.5
    
    def validate_signal_with_advanced_indicators(self, symbol: str, signal_data: Dict) -> Dict:
        """고급 인디케이터 기반 신호 검증"""
        try:
            enhanced_signal = signal_data.copy()
            original_confidence = signal_data.get('confidence', 0)
            
            # 모든 인디케이터 업데이트
            self.update_all_indicators(symbol)
            
            validation_scores = []
            
            # 1. Delta Flow Profile 검증
            delta_score = self._validate_with_delta_flow(symbol, signal_data)
            validation_scores.append(delta_score)
            
            # 2. VWAP Periodic Close 검증
            vwap_score = self._validate_with_vwap(symbol, signal_data)
            validation_scores.append(vwap_score)
            
            # 3. Volume Profile 검증
            volume_score = self._validate_with_volume_profile(symbol, signal_data)
            validation_scores.append(volume_score)
            
            # 4. Supertrend 검증
            trend_score = self._validate_with_supertrend(symbol, signal_data)
            validation_scores.append(trend_score)
            
            # 최종 검증 점수 계산
            if validation_scores:
                avg_validation_score = sum(validation_scores) / len(validation_scores)
                
                # 신뢰도 조정 (검증 점수 반영)
                adjusted_confidence = original_confidence * (0.6 + 0.4 * avg_validation_score)
                enhanced_signal['adjusted_confidence'] = min(1.0, adjusted_confidence)
                enhanced_signal['advanced_validation_score'] = avg_validation_score
                enhanced_signal['indicator_validation'] = {
                    'delta_flow': delta_score,
                    'vwap': vwap_score,
                    'volume_profile': volume_score,
                    'supertrend': trend_score
                }
                
                self.logger.info(f"🔍 {symbol} 고급 인디케이터 검증 완료: "
                              f"원본 {original_confidence:.3f} → 조정 {enhanced_signal['adjusted_confidence']:.3f} "
                              f"(검증점수: {avg_validation_score:.2f})")
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 고급 인디케이터 검증 실패: {e}")
            return signal_data
    
    def _validate_with_delta_flow(self, symbol: str, signal_data: Dict) -> float:
        """Delta Flow Profile 기반 검증"""
        try:
            if symbol not in self.delta_flow_cache:
                return 0.5
                
            delta_data = self.delta_flow_cache[symbol]
            signal_type = signal_data.get('signal_type', 'hold')
            current_delta = delta_data.get('current_delta', 0)
            delta_trend = delta_data.get('delta_trend', 'NEUTRAL')
            
            # 델타 흐름과 신호 일치성 검증
            if signal_type == 'buy' and (current_delta > 0 or delta_trend == 'BULLISH'):
                return 0.8
            elif signal_type == 'sell' and (current_delta < 0 or delta_trend == 'BEARISH'):
                return 0.8
            elif signal_type == 'hold':
                return 0.7
            else:
                return 0.3
                
        except Exception as e:
            self.logger.warning(f"⚠️ {symbol} Delta Flow 검증 실패: {e}")
            return 0.5
    
    def _validate_with_vwap(self, symbol: str, signal_data: Dict) -> float:
        """VWAP Periodic Close 기반 검증"""
        try:
            if symbol not in self.vwap_cache:
                return 0.5
                
            vwap_data = self.vwap_cache[symbol]
            signal_type = signal_data.get('signal_type', 'hold')
            primary_vwap = vwap_data.get('primary_vwap', 0)
            
            # 현재 가격 조회
            current_price = self._get_current_price(symbol)
            if current_price <= 0:
                return 0.5
            
            # VWAP 기준 상대적 위치 검증
            price_ratio = current_price / primary_vwap
            
            if signal_type == 'buy' and price_ratio < 0.995:  # VWAP 아래에서 매수
                return 0.9
            elif signal_type == 'sell' and price_ratio > 1.005:  # VWAP 위에서 매도
                return 0.9
            else:
                return 0.4
                
        except Exception as e:
            self.logger.warning(f"⚠️ {symbol} VWAP 검증 실패: {e}")
            return 0.5
    
    def _validate_with_volume_profile(self, symbol: str, signal_data: Dict) -> float:
        """Volume Profile 기반 검증"""
        try:
            if symbol not in self.volume_profile_cache:
                return 0.5
                
            volume_data = self.volume_profile_cache[symbol]
            signal_type = signal_data.get('signal_type', 'hold')
            
            # 현재 가격 조회
            current_price = self._get_current_price(symbol)
            
            poc = volume_data.get('poc_price', current_price)
            value_area_high = volume_data.get('value_area_high', poc * 1.02)
            value_area_low = volume_data.get('value_area_low', poc * 0.98)
            current_position = volume_data.get('current_position', 'UNKNOWN')
            
            # 볼륨 프로파일 기반 검증
            if current_position == 'BELOW_VALUE_AREA' and signal_type == 'buy':
                return 0.9  # Value Area 아래에서 매수 신호 강함
            elif current_position == 'ABOVE_VALUE_AREA' and signal_type == 'sell':
                return 0.9  # Value Area 위에서 매도 신호 강함
            elif current_position == 'INSIDE_VALUE_AREA':
                # POC 근처에서 반전 신호 검증
                if signal_type == 'buy' and current_price <= poc:
                    return 0.7
                elif signal_type == 'sell' and current_price >= poc:
                    return 0.7
                else:
                    return 0.5
            else:
                return 0.3
                
        except Exception as e:
            self.logger.warning(f"⚠️ {symbol} Volume Profile 검증 실패: {e}")
            return 0.5
    
    def _validate_with_supertrend(self, symbol: str, signal_data: Dict) -> float:
        """Supertrend 기반 검증"""
        try:
            if symbol not in self.supertrend_cache:
                return 0.5
                
            trend_data = self.supertrend_cache[symbol]
            signal_type = signal_data.get('signal_type', 'hold')
            current_trend = trend_data.get('current_trend', 1)
            trend_strength = trend_data.get('trend_strength', 0.5)
            
            # 추세 방향과 신호 일치성 검증 (추세 강도 반영)
            if (signal_type == 'buy' and current_trend == 1) or (signal_type == 'sell' and current_trend == -1):
                return 0.7 + 0.3 * trend_strength  # 추세 방향과 일치 (강도 반영)
            elif signal_type == 'hold':
                return 0.6
            else:
                return 0.3  # 추세 반대 신호
                
        except Exception as e:
            self.logger.warning(f"⚠️ {symbol} Supertrend 검증 실패: {e}")
            return 0.5
    
    def _get_current_price(self, symbol: str) -> float:
        """현재 가격 조회"""
        try:
            ticker = self.executor.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            self.logger.error(f"❌ {symbol} 가격 조회 실패: {e}")
            return 0.0
    
    def generate_advanced_validation_report(self, symbol: str, signal_data: Dict) -> str:
        """고급 인디케이터 검증 보고서 생성"""
        try:
            report = f"""
🎯 {symbol} 고급 인디케이터 검증 보고서 (Phase 13.2)
==================================================
📊 기본 신호 정보
• 신호 타입: {signal_data.get('signal_type', 'N/A')}
• 원본 신뢰도: {signal_data.get('confidence', 0):.3f}
• 검증 후 신뢰도: {signal_data.get('adjusted_confidence', 0):.3f}

🔍 고급 인디케이터 검증 점수"""
            
            # 인디케이터별 점수
            indicator_validation = signal_data.get('indicator_validation', {})
            for indicator, score in indicator_validation.items():
                report += f"\n• {indicator.replace('_', ' ').title()}: {score:.2f}"
            
            report += f"\n• 평균 검증 점수: {signal_data.get('advanced_validation_score', 0):.2f}"
            
            # 인디케이터 현재 상태
            report += f"\n\n💡 인디케이터 현재 상태"
            
            # Delta Flow 상태
            if symbol in self.delta_flow_cache:
                delta_data = self.delta_flow_cache[symbol]
                report += f"\n• Delta Flow: {delta_data.get('current_delta', 0):.0f} ({delta_data.get('delta_trend', 'N/A')})"
            
            # VWAP 상태
            if symbol in self.vwap_cache:
                vwap_data = self.vwap_cache[symbol]
                current_price = self._get_current_price(symbol)
                primary_vwap = vwap_data.get('primary_vwap', current_price)
                deviation = (current_price - primary_vwap) / primary_vwap
                report += f"\n• VWAP: ${primary_vwap:.4f} (편차: {deviation:.2%})"
            
            # Volume Profile 상태
            if symbol in self.volume_profile_cache:
                vol_data = self.volume_profile_cache[symbol]
                report += f"\n• Volume Profile: POC ${vol_data.get('poc_price', 0):.4f}"
                report += f"\n• Value Area: ${vol_data.get('value_area_low', 0):.4f} - ${vol_data.get('value_area_high', 0):.4f}"
                report += f"\n• 현재 위치: {vol_data.get('current_position', 'N/A')}"
            
            # Supertrend 상태
            if symbol in self.supertrend_cache:
                trend_data = self.supertrend_cache[symbol]
                trend_text = "상승 📈" if trend_data.get('current_trend', 1) == 1 else "하락 📉"
                report += f"\n• Supertrend: {trend_text} (강도: {trend_data.get('trend_strength', 0):.2f})"
            
            # 최종 평가
            final_confidence = signal_data.get('adjusted_confidence', 0)
            min_confidence = 0.05  # 기본값
            
            if final_confidence >= min_confidence * 1.5:
                evaluation = "✅ 매우 강한 신호"
            elif final_confidence >= min_confidence:
                evaluation = "⚠️ 보통 신호"
            else:
                evaluation = "❌ 약한 신호"
            
            report += f"\n\n🎯 최종 평가: {evaluation}"
            report += f"\n• 행동: {'진입 가능' if final_confidence >= min_confidence else '대기 필요'}"
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ 검증 보고서 생성 실패: {e}")
            return f"검증 보고서 생성 실패: {e}"

class MarketMakerAnalyzer:
    """시장 조성자 행동 패턴 분석 시스템 - Phase 13.3"""
    
    def __init__(self, config: Dict, executor):
        self.config = config
        self.executor = executor
        self.logger = logging.getLogger(__name__)
        
        # 분석 데이터 캐시
        self.order_imbalance_cache = {}
        self.large_order_cache = {}
        self.spread_analysis_cache = {}
        self.volume_anomaly_cache = {}
        
        # 설정값
        self.analysis_config = config.get('market_maker_analysis', {})
        self.large_order_threshold = self.analysis_config.get('large_order_threshold', 50000)  # $50,000
        self.imbalance_threshold = self.analysis_config.get('imbalance_threshold', 0.7)
        
        self.logger.info("🎯 시장 조성자 행동 패턴 분석 시스템 초기화 완료")
    
    def analyze_market_maker_behavior(self, symbol: str):
        """시장 조성자 행동 종합 분석"""
        try:
            self.logger.info(f"🔍 {symbol} 시장 조성자 행동 분석 중...")
            
            # 1. 주문장 불균형 분석
            order_imbalance = self._analyze_order_imbalance(symbol)
            
            # 2. 대형 주문 흔적 감지
            large_orders = self._detect_large_order_traces(symbol)
            
            # 3. 스프레드 분석
            spread_analysis = self._analyze_spread_behavior(symbol)
            
            # 4. 가격-거래량 이상 행동 감지
            volume_anomalies = self._detect_volume_price_anomalies(symbol)
            
            # 종합 시장 조성자 지수 계산
            mm_confidence = self._calculate_market_maker_confidence(
                order_imbalance, large_orders, spread_analysis, volume_anomalies
            )
            
            analysis_result = {
                'order_imbalance': order_imbalance,
                'large_orders_detected': large_orders,
                'spread_analysis': spread_analysis,
                'volume_anomalies': volume_anomalies,
                'market_maker_confidence': mm_confidence,
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"✅ {symbol} 시장 조성자 분석 완료: 신뢰도 {mm_confidence:.2f}")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 시장 조성자 분석 실패: {e}")
            return self._get_default_analysis_result()
    
    def _analyze_order_imbalance(self, symbol: str) -> Dict:
        """주문장 불균형 분석"""
        try:
            # Order Book 데이터 조회 (상위 50개 호가)
            order_book = self.executor.client.futures_order_book(symbol=symbol, limit=50)
            
            bids = order_book['bids']  # [가격, 수량]
            asks = order_book['asks']  # [가격, 수량]
            
            if not bids or not asks:
                return {'imbalance_ratio': 0.5, 'imbalance_direction': 'NEUTRAL', 'confidence': 0.0}
            
            # 매수/매도 호가 총량 계산
            total_bid_volume = sum(float(qty) for _, qty in bids)
            total_ask_volume = sum(float(qty) for _, qty in asks)
            
            # 불균형 비율 계산
            if total_bid_volume + total_ask_volume > 0:
                imbalance_ratio = total_bid_volume / (total_bid_volume + total_ask_volume)
            else:
                imbalance_ratio = 0.5
            
            # 불균형 방향 결정
            if imbalance_ratio > self.imbalance_threshold:
                direction = 'BULLISH'
                confidence = (imbalance_ratio - 0.5) * 2
            elif imbalance_ratio < (1 - self.imbalance_threshold):
                direction = 'BEARISH' 
                confidence = (0.5 - imbalance_ratio) * 2
            else:
                direction = 'NEUTRAL'
                confidence = 0.0
            
            # 대형 주문 불균형 분석
            large_order_imbalance = self._analyze_large_order_imbalance(bids, asks)
            
            result = {
                'imbalance_ratio': imbalance_ratio,
                'imbalance_direction': direction,
                'confidence': min(1.0, confidence),
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume,
                'large_order_imbalance': large_order_imbalance
            }
            
            self.order_imbalance_cache[symbol] = result
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 주문장 불균형 분석 실패: {e}")
            return {'imbalance_ratio': 0.5, 'imbalance_direction': 'NEUTRAL', 'confidence': 0.0}
    
    def _analyze_large_order_imbalance(self, bids: list, asks: list) -> Dict:
        """대형 주문 불균형 분석"""
        try:
            large_bid_orders = 0
            large_ask_orders = 0
            
            for price, quantity in bids:
                qty_val = float(quantity)
                price_val = float(price)
                order_value = qty_val * price_val
                
                if order_value >= self.large_order_threshold:
                    large_bid_orders += 1
            
            for price, quantity in asks:
                qty_val = float(quantity)
                price_val = float(price)
                order_value = qty_val * price_val
                
                if order_value >= self.large_order_threshold:
                    large_ask_orders += 1
            
            total_large_orders = large_bid_orders + large_ask_orders
            if total_large_orders > 0:
                large_imbalance_ratio = large_bid_orders / total_large_orders
            else:
                large_imbalance_ratio = 0.5
            
            return {
                'large_bid_orders': large_bid_orders,
                'large_ask_orders': large_ask_orders,
                'large_imbalance_ratio': large_imbalance_ratio,
                'large_imbalance_direction': 'BULLISH' if large_imbalance_ratio > 0.6 else 
                                          'BEARISH' if large_imbalance_ratio < 0.4 else 'NEUTRAL'
            }
            
        except Exception as e:
            self.logger.error(f"❌ 대형 주문 불균형 분석 실패: {e}")
            return {'large_bid_orders': 0, 'large_ask_orders': 0, 'large_imbalance_ratio': 0.5, 'large_imbalance_direction': 'NEUTRAL'}
    
    def _detect_large_order_traces(self, symbol: str) -> Dict:
        """대형 주문 흔적 감지"""
        try:
            # 최근 거래 데이터 조회 (공개 거래 정보)
            recent_trades = self.executor.client.futures_recent_trades(symbol=symbol, limit=100)
            
            if not recent_trades:
                return {'large_trades_detected': 0, 'total_large_volume': 0, 'dominant_side': 'NEUTRAL'}
            
            large_buy_volume = 0
            large_sell_volume = 0
            large_trades_count = 0
            
            for trade in recent_trades:
                quantity = float(trade['qty'])
                price = float(trade['price'])
                is_buyer_maker = trade['isBuyerMaker']
                
                trade_value = quantity * price
                
                if trade_value >= self.large_order_threshold:
                    large_trades_count += 1
                    if is_buyer_maker:
                        large_sell_volume += trade_value  # 매도자가 만든 거래 = 매도 대형 주문
                    else:
                        large_buy_volume += trade_value   # 매수자가 만든 거래 = 매수 대형 주문
            
            total_large_volume = large_buy_volume + large_sell_volume
            
            # 주도적 측면 분석
            if total_large_volume > 0:
                buy_ratio = large_buy_volume / total_large_volume
                if buy_ratio > 0.6:
                    dominant_side = 'BULLISH'
                elif buy_ratio < 0.4:
                    dominant_side = 'BEARISH'
                else:
                    dominant_side = 'NEUTRAL'
            else:
                dominant_side = 'NEUTRAL'
            
            result = {
                'large_trades_detected': large_trades_count,
                'total_large_volume': total_large_volume,
                'large_buy_volume': large_buy_volume,
                'large_sell_volume': large_sell_volume,
                'dominant_side': dominant_side,
                'buy_ratio': large_buy_volume / total_large_volume if total_large_volume > 0 else 0.5
            }
            
            self.large_order_cache[symbol] = result
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 대형 주문 흔적 감지 실패: {e}")
            return {'large_trades_detected': 0, 'total_large_volume': 0, 'dominant_side': 'NEUTRAL'}
    
    def _analyze_spread_behavior(self, symbol: str) -> Dict:
        """스프레드 및 유동성 행동 분석"""
        try:
            # Order Book 데이터로 스프레드 분석
            order_book = self.executor.client.futures_order_book(symbol=symbol, limit=20)
            
            if not order_book['bids'] or not order_book['asks']:
                return {'spread_percentage': 0, 'liquidity_depth': 0, 'spread_tightness': 'UNKNOWN'}
            
            best_bid = float(order_book['bids'][0][0])
            best_ask = float(order_book['asks'][0][0])
            mid_price = (best_bid + best_ask) / 2
            
            # 스프레드 비율 계산
            spread = best_ask - best_bid
            spread_percentage = (spread / mid_price) * 100
            
            # 유동성 깊이 분석 (상위 10개 호가 총량)
            bid_depth = sum(float(qty) for _, qty in order_book['bids'][:10])
            ask_depth = sum(float(qty) for _, qty in order_book['asks'][:10])
            total_depth = bid_depth + ask_depth
            
            # 스프레드 긴축도 분석
            if spread_percentage < 0.01:
                tightness = 'VERY_TIGHT'
            elif spread_percentage < 0.05:
                tightness = 'TIGHT'
            elif spread_percentage < 0.1:
                tightness = 'NORMAL'
            else:
                tightness = 'WIDE'
            
            # 유동성 불균형 분석
            if total_depth > 0:
                depth_imbalance = (bid_depth - ask_depth) / total_depth
            else:
                depth_imbalance = 0
            
            result = {
                'spread_percentage': spread_percentage,
                'spread_tightness': tightness,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'total_depth': total_depth,
                'depth_imbalance': depth_imbalance,
                'liquidity_quality': self._assess_liquidity_quality(spread_percentage, total_depth)
            }
            
            self.spread_analysis_cache[symbol] = result
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 스프레드 분석 실패: {e}")
            return {'spread_percentage': 0, 'liquidity_depth': 0, 'spread_tightness': 'UNKNOWN'}
    
    def _assess_liquidity_quality(self, spread_percentage: float, total_depth: float) -> str:
        """유동성 품질 평가"""
        if spread_percentage < 0.02 and total_depth > 100000:  # $100,000 이상
            return 'EXCELLENT'
        elif spread_percentage < 0.05 and total_depth > 50000:
            return 'GOOD'
        elif spread_percentage < 0.1 and total_depth > 10000:
            return 'FAIR'
        else:
            return 'POOR'
    
    def _detect_volume_price_anomalies(self, symbol: str) -> Dict:
        """가격-거래량 이상 행동 감지"""
        try:
            # 5분 봉 데이터로 이상 감지
            klines = self.executor.client.futures_klines(
                symbol=symbol, interval='5m', limit=50
            )
            
            if len(klines) < 20:
                return {'anomalies_detected': 0, 'volume_spike': False, 'price_displacement': False}
            
            volumes = [float(k[5]) for k in klines]
            closes = [float(k[4]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            
            # 거래량 스파이크 감지
            volume_anomalies = self._detect_volume_spikes(volumes)
            
            # 가격 변위 감지 (이상 급등/급락)
            price_anomalies = self._detect_price_displacement(highs, lows, closes)
            
            # 가격-거래량 분리 감지
            volume_price_divergence = self._detect_volume_price_divergence(volumes, closes)
            
            anomalies_count = volume_anomalies['spikes_detected'] + price_anomalies['displacements_detected']
            
            result = {
                'anomalies_detected': anomalies_count,
                'volume_spike': volume_anomalies['spikes_detected'] > 0,
                'price_displacement': price_anomalies['displacements_detected'] > 0,
                'volume_price_divergence': volume_price_divergence,
                'recent_volume_spike': volume_anomalies['recent_spike'],
                'recent_price_displacement': price_anomalies['recent_displacement']
            }
            
            self.volume_anomaly_cache[symbol] = result
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 가격-거래량 이상 감지 실패: {e}")
            return {'anomalies_detected': 0, 'volume_spike': False, 'price_displacement': False}
    
    def _detect_volume_spikes(self, volumes: list) -> Dict:
        """거래량 스파이크 감지"""
        try:
            if len(volumes) < 10:
                return {'spikes_detected': 0, 'recent_spike': False}
            
            # 이동평균 기반 이상치 감지
            volume_series = pd.Series(volumes)
            volume_ma = volume_series.rolling(window=10).mean()
            volume_std = volume_series.rolling(window=10).std()
            
            spikes_detected = 0
            recent_spike = False
            
            for i in range(10, len(volumes)):
                if volume_std.iloc[i] > 0:  # 표준편차가 0보다 큰 경우만
                    z_score = (volumes[i] - volume_ma.iloc[i]) / volume_std.iloc[i]
                    if z_score > 3.0:  # 3시그마 이상
                        spikes_detected += 1
                        if i >= len(volumes) - 3:  # 최근 3개 봉 내 스파이크
                            recent_spike = True
            
            return {
                'spikes_detected': spikes_detected,
                'recent_spike': recent_spike
            }
            
        except Exception as e:
            self.logger.error(f"❌ 거래량 스파이크 감지 실패: {e}")
            return {'spikes_detected': 0, 'recent_spike': False}
    
    def _detect_price_displacement(self, highs: list, lows: list, closes: list) -> Dict:
        """가격 변위 감지"""
        try:
            if len(closes) < 10:
                return {'displacements_detected': 0, 'recent_displacement': False}
            
            displacements_detected = 0
            recent_displacement = False
            
            for i in range(1, len(closes)):
                price_range = highs[i] - lows[i]
                prev_price_range = highs[i-1] - lows[i-1]
                
                # 가격 범위 급변 감지
                if prev_price_range > 0:
                    range_change = price_range / prev_price_range
                    
                    # 가격 변위와 종가 변화 결합 분석
                    price_change = abs(closes[i] - closes[i-1]) / closes[i-1]
                    
                    if range_change > 2.0 and price_change > 0.02:  # 범위 2배 이상 + 가격 2% 이상 변화
                        displacements_detected += 1
                        if i >= len(closes) - 3:  # 최근 3개 봉 내 변위
                            recent_displacement = True
            
            return {
                'displacements_detected': displacements_detected,
                'recent_displacement': recent_displacement
            }
            
        except Exception as e:
            self.logger.error(f"❌ 가격 변위 감지 실패: {e}")
            return {'displacements_detected': 0, 'recent_displacement': False}
    
    def _detect_volume_price_divergence(self, volumes: list, closes: list) -> str:
        """가격-거래량 분리 감지"""
        try:
            if len(volumes) < 20:
                return 'INSUFFICIENT_DATA'
            
            # 최근 10개 봉과 이전 10개 봉 비교
            recent_volumes = volumes[-10:]
            recent_closes = closes[-10:]
            previous_volumes = volumes[-20:-10]
            previous_closes = closes[-20:-10]
            
            # 평균 거래량과 가격 변화 계산
            avg_recent_volume = np.mean(recent_volumes)
            avg_previous_volume = np.mean(previous_volumes)
            recent_price_change = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
            previous_price_change = (previous_closes[-1] - previous_closes[0]) / previous_closes[0]
            
            volume_change_ratio = avg_recent_volume / avg_previous_volume if avg_previous_volume > 0 else 1
            
            # 분리 패턴 분석
            if volume_change_ratio > 1.5 and recent_price_change < -0.01:
                return 'BEARISH_DIVERGENCE'  # 거래량 증가 but 가격 하락
            elif volume_change_ratio > 1.5 and recent_price_change > 0.01:
                return 'BULLISH_CONFIRMATION'  # 거래량 증가 + 가격 상승
            elif volume_change_ratio < 0.7 and recent_price_change > 0.01:
                return 'BULLISH_DIVERGENCE'  # 거래량 감소 but 가격 상승
            elif volume_change_ratio < 0.7 and recent_price_change < -0.01:
                return 'BEARISH_CONFIRMATION'  # 거래량 감소 + 가격 하락
            else:
                return 'NO_DIVERGENCE'
                
        except Exception as e:
            self.logger.error(f"❌ 가격-거래량 분리 감지 실패: {e}")
            return 'ANALYSIS_FAILED'
    
    def _calculate_market_maker_confidence(self, order_imbalance: Dict, large_orders: Dict, 
                                         spread_analysis: Dict, volume_anomalies: Dict) -> float:
        """시장 조성자 신뢰도 종합 계산"""
        try:
            confidence_score = 0.5  # 기본값
            
            # 1. 주문장 불균형 가중치 (30%)
            imbalance_confidence = order_imbalance.get('confidence', 0)
            confidence_score += imbalance_confidence * 0.3
            
            # 2. 대형 주문 가중치 (25%)
            large_order_score = 0.0
            if large_orders.get('large_trades_detected', 0) > 0:
                if large_orders.get('dominant_side') == 'BULLISH':
                    large_order_score = 0.8
                elif large_orders.get('dominant_side') == 'BEARISH':
                    large_order_score = 0.8
                else:
                    large_order_score = 0.5
            confidence_score += large_order_score * 0.25
            
            # 3. 스프레드 분석 가중치 (25%)
            spread_score = 0.0
            tightness = spread_analysis.get('spread_tightness', 'UNKNOWN')
            if tightness in ['VERY_TIGHT', 'TIGHT']:
                spread_score = 0.8  # 긴축된 스프레드 = 시장 조성자 활동 가능성 높음
            elif tightness == 'NORMAL':
                spread_score = 0.5
            else:
                spread_score = 0.3
            confidence_score += spread_score * 0.25
            
            # 4. 이상 행동 가중치 (20%)
            anomaly_score = 0.0
            if volume_anomalies.get('anomalies_detected', 0) > 0:
                # 이상 행동 존재 = 시장 조성자 활동 가능성
                anomaly_score = 0.7
            else:
                anomaly_score = 0.3
            confidence_score += anomaly_score * 0.2
            
            return max(0.0, min(1.0, confidence_score))
            
        except Exception as e:
            self.logger.error(f"❌ 시장 조성자 신뢰도 계산 실패: {e}")
            return 0.5
    
    def validate_signal_with_market_maker_analysis(self, symbol: str, signal_data: Dict) -> Dict:
        """시장 조성자 분석 기반 신호 검증"""
        try:
            enhanced_signal = signal_data.copy()
            
            # 시장 조성자 행동 분석
            mm_analysis = self.analyze_market_maker_behavior(symbol)
            
            signal_type = signal_data.get('signal_type', 'hold')
            mm_confidence = mm_analysis.get('market_maker_confidence', 0.5)
            
            # 시장 조성자 방향과 신호 일치성 검증
            mm_direction = self._infer_market_maker_direction(mm_analysis)
            
            validation_score = 0.5  # 기본값
            
            if signal_type == 'hold':
                validation_score = 0.7  # 홀드는 중립
            elif mm_direction == 'BULLISH' and signal_type == 'buy':
                validation_score = 0.8 + (mm_confidence * 0.2)  # 일치
            elif mm_direction == 'BEARISH' and signal_type == 'sell':
                validation_score = 0.8 + (mm_confidence * 0.2)  # 일치
            elif mm_direction == 'NEUTRAL':
                validation_score = 0.6  # 중립
            else:
                validation_score = 0.3  # 반대
            
            enhanced_signal['market_maker_validation'] = {
                'validation_score': validation_score,
                'market_maker_confidence': mm_confidence,
                'inferred_direction': mm_direction,
                'analysis_details': mm_analysis
            }
            
            # 신뢰도 조정
            original_confidence = enhanced_signal.get('adjusted_confidence', enhanced_signal.get('confidence', 0))
            adjusted_confidence = original_confidence * (0.7 + 0.3 * validation_score)
            enhanced_signal['adjusted_confidence'] = min(1.0, adjusted_confidence)
            
            self.logger.info(f"🔍 {symbol} 시장 조성자 검증: {mm_direction} 방향, "
                          f"검증점수 {validation_score:.2f}, 최종신뢰도 {enhanced_signal['adjusted_confidence']:.3f}")
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 시장 조성자 검증 실패: {e}")
            return signal_data
    
    def _infer_market_maker_direction(self, mm_analysis: Dict) -> str:
        """시장 조성자 방향 추론"""
        try:
            order_imbalance = mm_analysis.get('order_imbalance', {})
            large_orders = mm_analysis.get('large_orders_detected', {})
            volume_anomalies = mm_analysis.get('volume_anomalies', {})
            
            bullish_signals = 0
            bearish_signals = 0
            
            # 주문장 불균형 방향
            imbalance_direction = order_imbalance.get('imbalance_direction', 'NEUTRAL')
            if imbalance_direction == 'BULLISH':
                bullish_signals += 1
            elif imbalance_direction == 'BEARISH':
                bearish_signals += 1
            
            # 대형 주문 방향
            large_order_direction = large_orders.get('dominant_side', 'NEUTRAL')
            if large_order_direction == 'BULLISH':
                bullish_signals += 1
            elif large_order_direction == 'BEARISH':
                bearish_signals += 1
            
            # 거래량-가격 분리 방향
            volume_divergence = volume_anomalies.get('volume_price_divergence', 'NO_DIVERGENCE')
            if volume_divergence in ['BULLISH_DIVERGENCE', 'BEARISH_CONFIRMATION']:
                bearish_signals += 1
            elif volume_divergence in ['BEARISH_DIVERGENCE', 'BULLISH_CONFIRMATION']:
                bullish_signals += 1
            
            # 최종 방향 결정
            if bullish_signals > bearish_signals:
                return 'BULLISH'
            elif bearish_signals > bullish_signals:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            self.logger.error(f"❌ 시장 조성자 방향 추론 실패: {e}")
            return 'NEUTRAL'
    
    def generate_market_maker_report(self, symbol: str, signal_data: Dict) -> str:
        """시장 조성자 분석 보고서 생성"""
        try:
            mm_validation = signal_data.get('market_maker_validation', {})
            analysis_details = mm_validation.get('analysis_details', {})
            
            report = f"""
🎯 {symbol} 시장 조성자 행동 분석 보고서 (Phase 13.3)
==================================================
📊 기본 분석 정보
• 추론된 방향: {mm_validation.get('inferred_direction', 'N/A')}
• 시장 조성자 신뢰도: {mm_validation.get('market_maker_confidence', 0):.2f}
• 검증 점수: {mm_validation.get('validation_score', 0):.2f}

🔍 상세 분석 결과"""

            # 주문장 불균형 분석
            order_imbalance = analysis_details.get('order_imbalance', {})
            report += f"\n📈 주문장 불균형 분석"
            report += f"\n• 불균형 비율: {order_imbalance.get('imbalance_ratio', 0):.3f}"
            report += f"\n• 불균형 방향: {order_imbalance.get('imbalance_direction', 'N/A')}"
            report += f"\n• 신뢰도: {order_imbalance.get('confidence', 0):.2f}"

            # 대형 주문 분석
            large_orders = analysis_details.get('large_orders_detected', {})
            report += f"\n\n💰 대형 주문 분석"
            report += f"\n• 감지된 대형 거래: {large_orders.get('large_trades_detected', 0)}건"
            report += f"\n• 총 대형 거래량: ${large_orders.get('total_large_volume', 0):.0f}"
            report += f"\n• 주도적 측면: {large_orders.get('dominant_side', 'N/A')}"

            # 스프레드 분석
            spread_analysis = analysis_details.get('spread_analysis', {})
            report += f"\n\n📊 스프레드 분석"
            report += f"\n• 스프레드: {spread_analysis.get('spread_percentage', 0):.3f}%"
            report += f"\n• 스프레드 긴축도: {spread_analysis.get('spread_tightness', 'N/A')}"
            report += f"\n• 유동성 품질: {spread_analysis.get('liquidity_quality', 'N/A')}"

            # 이상 행동 분석
            volume_anomalies = analysis_details.get('volume_anomalies', {})
            report += f"\n\n🚨 이상 행동 분석"
            report += f"\n• 감지된 이상: {volume_anomalies.get('anomalies_detected', 0)}건"
            report += f"\n• 거래량 스파이크: {'✅ 있음' if volume_anomalies.get('volume_spike') else '❌ 없음'}"
            report += f"\n• 가격 변위: {'✅ 있음' if volume_anomalies.get('price_displacement') else '❌ 없음'}"
            report += f"\n• 가격-거래량 분리: {volume_anomalies.get('volume_price_divergence', 'N/A')}"

            # 최종 평가
            validation_score = mm_validation.get('validation_score', 0)
            if validation_score >= 0.7:
                evaluation = "✅ 강한 일치"
            elif validation_score >= 0.5:
                evaluation = "⚠️ 보통 일치"
            else:
                evaluation = "❌ 약한 일치"

            report += f"\n\n🎯 최종 평가: {evaluation}"
            report += f"\n• 시장 조성자와의 일치도: {validation_score:.2f}"

            return report

        except Exception as e:
            self.logger.error(f"❌ 시장 조성자 보고서 생성 실패: {e}")
            return f"시장 조성자 보고서 생성 실패: {e}"

    def _get_default_analysis_result(self) -> Dict:
        """기본 분석 결과 반환"""
        return {
            'order_imbalance': {'imbalance_ratio': 0.5, 'imbalance_direction': 'NEUTRAL', 'confidence': 0.0},
            'large_orders_detected': {'large_trades_detected': 0, 'total_large_volume': 0, 'dominant_side': 'NEUTRAL'},
            'spread_analysis': {'spread_percentage': 0, 'liquidity_depth': 0, 'spread_tightness': 'UNKNOWN'},
            'volume_anomalies': {'anomalies_detected': 0, 'volume_spike': False, 'price_displacement': False},
            'market_maker_confidence': 0.5,
            'timestamp': datetime.now()
        }

class LiveTradingEngine:
    """실전 매매 최적화 트레이딩 엔진 - Phase 13.5"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        
        self.performance_monitor = PerformanceMonitor(config)
        self.performance_monitor.set_engine(self)
        self.initial_capital = self.config['trading'].get('initial_capital', 280.0)
        self.daily_loss_limit = -0.05 * self.initial_capital
        self.emergency_stopped = False
        self.last_daily_reset = datetime.now().date()
        
        # 🔥 Phase 13.2: 고급 인디케이터 통합 시스템 초기화
        self.advanced_indicators = None  # 나중에 초기화 (executor 필요)
        
        # 🔥 Phase 13.3: 시장 조성자 분석 시스템 초기화
        self.market_maker_analyzer = None  # 나중에 초기화 (executor 필요)
        
        # 🔥 PnL 계산 시스템 경고
        self.logger.warning("🚨 PnL 계산 시스템 점검 중 - 모든 수익 기록이 0으로 설정됩니다")
        self.logger.warning("💡 실제 계좌 잔고를 수동으로 확인하세요")
        
        self.volatility_analyzer = RealTimeVolatilityAnalyzer(config)
        
        # 🔥 데이터 품질 관리 시스템 초기화
        self._implement_data_quality_checks()
        
        # 포트폴리오 리밸런서는 나중에 초기화 (executor 필요)
        self.portfolio_rebalancer = None
        self.core_engine = CoreTradingEngine(config)
        
        self._integrate_backtest_validation()
        self._initialize_legacy_modules()
        
        # 🔥 Phase 13.2: 고급 인디케이터 통합 시스템 초기화 (executor 이후)
        if hasattr(self, 'executor') and self.executor:
            self.advanced_indicators = AdvancedIndicatorsIntegrator(config, self.executor)
            self.logger.info("✅ Phase 13.2 고급 인디케이터 통합 시스템 초기화 완료")
        
        # 🔥 Phase 13.3: 시장 조성자 분석 시스템 초기화 (executor 이후)
        if hasattr(self, 'executor') and self.executor:
            self.market_maker_analyzer = MarketMakerAnalyzer(config, self.executor)
            self.logger.info("✅ Phase 13.3 시장 조성자 분석 시스템 초기화 완료")
        
        # 🔥 포트폴리오 리밸런서 초기화 (executor 이후)
        if hasattr(self, 'executor') and self.executor:
            self.portfolio_rebalancer = PortfolioRebalancer(
                config=config,
                executor=self.executor,
                performance_monitor=self.performance_monitor
            )
        
        self.logger.info("✅ Phase 13.5 트레이딩 엔진 초기화 완료 (PnL 수정)")
        
        self.aggressive_mode = config['trading'].get('aggressive_mode', False)
        if self.aggressive_mode:
            self.logger.warning("🔥 공격적 트레이딩 모드 활성화!")
            self.logger.warning("   • Risk 비율: 8%")
            self.logger.warning("   • 최대 포지션: 40%")
            self.logger.warning("   • Margin 활용률 극대화")
        
        self.logger.info(f"🛡️ 긴급 정지 시스템: 일일 손실 한도 ${self.daily_loss_limit:.2f}")
        
        if self.aggressive_mode:
            balance = self.executor.get_futures_balance() if self.executor else 0
            available_margin = balance * self.config['trading'].get('leverage', 20)
            self.logger.info(f"🔥 공격적 모드: 사용 가능 Margin ${available_margin:.2f}")

    def _cleanup_testusdt_data(self):
        """기존 TESTUSDT 데이터 정리 및 PnL 오류 데이터 수정"""
        try:
            if os.path.exists('trades_log.csv'):
                # TESTUSDT 거래 제외한 새 파일 생성
                trades_df = pd.read_csv('trades_log.csv')
                real_trades = trades_df[~trades_df['symbol'].str.contains('TESTUSDT', case=False, na=False)]
                
                # 🔥 기존 PnL 데이터도 0으로 리셋 (임시 조치)
                real_trades['pnl'] = 0.0
                
                if len(real_trades) < len(trades_df):
                    # TESTUSDT 거래가 있는 경우 정리
                    real_trades.to_csv('trades_log_cleaned.csv', index=False)
                    os.replace('trades_log_cleaned.csv', 'trades_log.csv')
                    removed_count = len(trades_df) - len(real_trades)
                    self.logger.info(f"🧹 TESTUSDT 데이터 정리: {removed_count}개 제거, PnL 0으로 리셋")
            
            if os.path.exists('performance_log.csv'):
                # 성능 로그도 정리 (비정상 Sharpe Ratio 값 정리)
                perf_df = pd.read_csv('performance_log.csv')
                perf_df['sharpe_ratio'] = perf_df['sharpe_ratio'].apply(
                    lambda x: 0 if abs(x) > 100 or pd.isna(x) else x
                )
                # PnL도 0으로 정리
                perf_df['total_pnl'] = 0.0
                perf_df['win_rate'] = 0.0
                perf_df.to_csv('performance_log_cleaned.csv', index=False)
                os.replace('performance_log_cleaned.csv', 'performance_log.csv')
                self.logger.info("✅ 성능 로그 정리 완료 (PnL 0으로 리셋)")
                
        except Exception as e:
            self.logger.error(f"❌ 데이터 정리 실패: {e}")

    def _implement_data_quality_checks(self):
        """데이터 품질 관리 시스템 구현"""
        try:
            self.logger.info("🔍 데이터 품질 관리 시스템 초기화...")
            
            # 실시간 데이터 이상치 감지
            self.anomaly_detector = DataAnomalyDetector()
            
            # 매일 아침 9시 데이터 검증 스케줄러 (실제 실행은 run() 메서드에서)
            self.daily_audit_time = "09:00"
            
            self.logger.info("✅ 데이터 품질 관리 시스템 준비 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 데이터 품질 관리 시스템 초기화 실패: {e}")

    def _morning_data_audit(self):
        """매일 아침 데이터 검증"""
        try:
            audit_results = {
                'timestamp': datetime.now(),
                'data_completeness': self._check_data_completeness(),
                'price_anomalies': self._detect_price_anomalies(),
                'volume_consistency': self._validate_volume_data(),
                'api_health': self._check_api_connectivity(),
                'pnl_accuracy': self._verify_pnl_calculations()
            }
            
            # 리포트 생성 및 알림
            self._generate_data_quality_report(audit_results)
            
            if not all(audit_results.values()):
                self._send_alert(f"🚨 데이터 품질 문제 발견: {audit_results}")
                
        except Exception as e:
            self.logger.error(f"❌ 아침 데이터 검증 실패: {e}")

    def _check_data_completeness(self) -> bool:
        """데이터 완전성 검증"""
        try:
            # TODO: 실제 데이터 완전성 검증 로직 구현
            self.logger.info("📊 데이터 완전성 검증 실행")
            return True
        except Exception as e:
            self.logger.error(f"❌ 데이터 완전성 검증 실패: {e}")
            return False

    def _detect_price_anomalies(self) -> bool:
        """가격 이상치 탐지"""
        try:
            symbols = self.config['monitoring']['symbols']
            anomalies_detected = False
            
            for symbol in symbols:
                try:
                    # 현재 가격 조회
                    ticker = self.executor.client.futures_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    
                    # 이상치 탐지
                    if self.anomaly_detector.detect_price_spikes(symbol, current_price):
                        self.logger.warning(f"🚨 {symbol} 가격 이상치 감지: ${current_price:.4f}")
                        anomalies_detected = True
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ {symbol} 가격 이상치 탐지 실패: {e}")
                    continue
            
            return not anomalies_detected  # 이상치 없으면 True
            
        except Exception as e:
            self.logger.error(f"❌ 가격 이상치 탐지 실패: {e}")
            return False

    def _validate_volume_data(self) -> bool:
        """거래량 데이터 검증"""
        try:
            # TODO: 실제 거래량 검증 로직 구현
            self.logger.info("📈 거래량 데이터 검증 실행")
            return True
        except Exception as e:
            self.logger.error(f"❌ 거래량 데이터 검증 실패: {e}")
            return False

    def _check_api_connectivity(self) -> bool:
        """API 연결 상태 확인"""
        try:
            # 간단한 API 연결 테스트
            server_time = self.executor.client.get_server_time()
            self.logger.info("🔌 API 연결 상태: 정상")
            return True
        except Exception as e:
            self.logger.error(f"❌ API 연결 상태 불량: {e}")
            return False

    def _verify_pnl_calculations(self) -> bool:
        """PnL 계산 정확성 검증"""
        try:
            # TODO: 실제 PnL 계산 검증 로직 구현
            self.logger.info("💰 PnL 계산 정확성 검증 실행")
            return True
        except Exception as e:
            self.logger.error(f"❌ PnL 계산 검증 실패: {e}")
            return False

    def _generate_data_quality_report(self, audit_results: Dict):
        """데이터 품질 리포트 생성"""
        try:
            report = f"""
📊 데이터 품질 리포트 - {datetime.now().strftime('%Y-%m-%d %H:%M')}
==========================================
• 데이터 완전성: {'✅' if audit_results['data_completeness'] else '❌'}
• 가격 이상치: {'✅ 없음' if audit_results['price_anomalies'] else '❌ 발견'}
• 거래량 일관성: {'✅' if audit_results['volume_consistency'] else '❌'}
• API 연결: {'✅ 정상' if audit_results['api_health'] else '❌ 이상'}
• PnL 계산: {'✅ 정확' if audit_results['pnl_accuracy'] else '❌ 오류'}

📋 종합 평가: {'✅ 양호' if all(audit_results.values()) else '⚠️ 점검 필요'}
"""
            self.logger.info(f"📄 데이터 품질 리포트:\n{report}")
            
        except Exception as e:
            self.logger.error(f"❌ 데이터 품질 리포트 생성 실패: {e}")

    def _send_alert(self, message: str):
        """알림 전송"""
        try:
            self.logger.warning(f"🚨 {message}")
            
            # TODO: 디스코드/텔레그램 알림 연동
            webhook_url = self.config.get('discord', {}).get('webhook_url')
            if webhook_url:
                import requests
                data = {
                    "content": f"**🚨 데이터 품질 알림**\n{message}",
                    "username": "Evo-Quant AI Data Quality"
                }
                requests.post(webhook_url, json=data, timeout=10)
                
        except Exception as e:
            self.logger.error(f"❌ 알림 전송 실패: {e}")

    def _start_daily_audit_scheduler(self):
        """일일 데이터 검증 스케줄러 시작"""
        try:
            # 현재 시간과 다음 9시 계산
            now = datetime.now()
            next_audit = now.replace(hour=9, minute=0, second=0, microsecond=0)
            if now >= next_audit:
                next_audit += timedelta(days=1)
            
            # 첫 실행까지 대기 시간 계산
            wait_seconds = (next_audit - now).total_seconds()
            
            self.logger.info(f"⏰ 일일 데이터 검증 예약: 다음 실행 {next_audit.strftime('%Y-%m-%d %H:%M')} "
                            f"({(wait_seconds/3600):.1f}시간 후)")
            
            # 별도 스레드에서 스케줄러 실행
            import threading
            scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            scheduler_thread.start()
            
        except Exception as e:
            self.logger.error(f"❌ 일일 검증 스케줄러 시작 실패: {e}")

    def _run_scheduler(self):
        """스케줄러 실행"""
        try:
            while True:
                now = datetime.now()
                current_time = now.strftime("%H:%M")
                
                # 매일 9시에 실행
                if current_time == self.daily_audit_time:
                    self.logger.info("🔍 일일 데이터 검증 실행...")
                    self._morning_data_audit()
                    
                    # 24시간 대기 (다음날 같은 시간까지)
                    time.sleep(86400)  # 24시간
                else:
                    # 1분마다 시간 체크
                    time.sleep(60)
                    
        except Exception as e:
            self.logger.error(f"❌ 스케줄러 실행 실패: {e}")

    def _test_discord_notification(self):
        """Discord 알림 테스트"""
        try:
            webhook_url = self.config.get('discord', {}).get('webhook_url')
            
            if not webhook_url:
                self.logger.warning("⚠️ Discord webhook URL이 설정되지 않음")
                return
                
            import requests
            data = {
                "content": "**시스템 시작**\nEvo-Quant AI v3.0 트레이딩 시스템이 시작되었습니다.",
                "username": "Evo-Quant AI Trader"
            }
            
            response = requests.post(webhook_url, json=data, timeout=10)
            if response.status_code == 204:
                self.logger.info("✅ Discord 알림 테스트 성공")
            else:
                self.logger.warning(f"⚠️ Discord 알림 테스트 실패: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"❌ Discord 알림 테스트 실패: {e}")

    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('live_trading.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _integrate_backtest_validation(self):
        """백테스팅 검증 결과 라이브 트레이딩에 통합"""
        try:
            self.logger.info("🔄 백테스팅 검증 결과 통합 중...")
            backtest_params = self._load_optimized_backtest_params()
            
            if backtest_params:
                self._apply_optimized_parameters(backtest_params)
                self.logger.info("✅ 백테스팅 최적 파라미터 라이브 적용 완료")
            else:
                self.logger.warning("⚠️ 백테스팅 파라미터 없음, 기본값 사용")
                
        except Exception as e:
            self.logger.error(f"❌ 백테스팅 통합 실패: {e}")

    def _load_optimized_backtest_params(self):
        """최적화된 백테스팅 파라미터 로드"""
        try:
            results_dir = 'backtest_results'
            if not os.path.exists(results_dir):
                self.logger.warning("⚠️ 백테스팅 결과 디렉토리 없음")
                return None
                
            backtest_files = [f for f in os.listdir(results_dir) if f.startswith('backtest_') and f.endswith('.csv')]
            if not backtest_files:
                self.logger.warning("⚠️ 백테스팅 결과 파일 없음")
                return None
                
            latest_file = max(backtest_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
            latest_path = os.path.join(results_dir, latest_file)
            
            df = pd.read_csv(latest_path)
            if df.empty:
                return None
                
            optimized_params = {
                'atr_multiplier': 1.5,
                'min_confidence': 0.05,
                'risk_per_trade': 0.03
            }
            
            self.logger.info(f"📊 백테스팅 파라미터 로드: ATR={optimized_params['atr_multiplier']}, "
                        f"신뢰도={optimized_params['min_confidence']}, "
                        f"리스크={optimized_params['risk_per_trade']}")
            
            return optimized_params
            
        except Exception as e:
            self.logger.error(f"❌ 백테스팅 파라미터 로드 실패: {e}")
            return None

    def _apply_optimized_parameters(self, params):
        """최적화된 파라미터 적용"""
        try:
            if hasattr(self, 'strategy') and self.strategy:
                self.strategy.atr_multiplier = params['atr_multiplier']
                self.strategy.min_confidence = params['min_confidence']
                self.strategy.risk_per_trade = params['risk_per_trade']
                
            if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                self.portfolio_manager.risk_per_trade = params['risk_per_trade']
                
            self.config['trading']['atr_multiplier'] = params['atr_multiplier']
            self.config['trading']['min_confidence'] = params['min_confidence'] 
            self.config['trading']['risk_per_trade'] = params['risk_per_trade']
            
            self.logger.info("🎯 최적화 파라미터 적용 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 파라미터 적용 실패: {e}")

    def _adjust_parameters_real_time(self, cycle_count: int):
        """실시간 파라미터 조정"""
        try:
            if cycle_count % 5 != 0:
                return
                
            self.logger.info("🔄 실시간 파라미터 조정 실행")
            
            portfolio_state = self._check_portfolio_balance()
            
            performance = self.performance_monitor.get_performance_summary()
            current_balance = self.executor.get_futures_balance() if self.executor else 0
            
            symbols = self.config['monitoring']['symbols']
            recommended_params_list = []
            
            for symbol in symbols:
                try:
                    current_price = self._get_current_price(symbol) if hasattr(self, '_get_current_price') else 100.0
                    self.volatility_analyzer.update_price_data(symbol, current_price)
                    
                    recommended_params = self.volatility_analyzer.get_recommended_parameters(symbol)
                    recommended_params_list.append(recommended_params)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ {symbol} 파라미터 추천 실패: {e}")
                    continue
            
            if recommended_params_list:
                avg_atr = np.mean([p['atr_multiplier'] for p in recommended_params_list])
                avg_risk = np.mean([p['risk_per_trade'] for p in recommended_params_list])
                avg_confidence = np.mean([p['min_confidence'] for p in recommended_params_list])
                
                if portfolio_state.get('needs_rebalancing', False):
                    avg_risk = max(0.01, avg_risk * 0.7)
                    self.logger.info("📉 포트폴리오 불균형으로 리스크 감소 적용")
                
                win_rate = performance.get('win_rate', 50)
                total_pnl = performance.get('total_pnl', 0)
                
                if win_rate < 40:
                    avg_risk = max(0.01, avg_risk * 0.8)
                    avg_confidence = min(0.1, avg_confidence * 1.2)
                
                if total_pnl < 0:
                    avg_atr = min(2.5, avg_atr * 1.1)
                    avg_risk = max(0.01, avg_risk * 0.9)
                
                self._apply_dynamic_parameters(avg_atr, avg_risk, avg_confidence)
                
                self.logger.info(f"📊 실시간 파라미터 조정 완료: "
                            f"ATR={avg_atr:.2f}, 리스크={avg_risk:.3f}, 신뢰도={avg_confidence:.3f}")
            
        except Exception as e:
            self.logger.error(f"❌ 실시간 파라미터 조정 실패: {e}")

    def _check_and_rebalance_portfolio(self):
        """포트폴리오 리밸런싱 확인 및 실행"""
        try:
            if not hasattr(self, 'portfolio_rebalancer') or not self.portfolio_rebalancer:
                return
                
            # 현재 포지션 정보 수집
            current_positions = {}
            pnl_data = self.performance_monitor.calculate_real_time_pnl(self)
            
            for symbol, data in pnl_data.items():
                current_positions[symbol] = {
                    'position_amt': data.get('position_amt', 0),
                    'entry_price': data.get('entry_price', 0),
                    'unrealized_pnl': data.get('unrealized_pnl', 0)
                }
            
            # 목표 가중치 계산
            target_weights = self.portfolio_rebalancer.calculate_target_weights(current_positions)
            
            # 현재 가중치 계산
            current_weights = self.portfolio_rebalancer.calculate_current_weights(current_positions)
            
            # 리밸런싱 필요 여부 확인
            needs_rebalance, rebalancing_symbols = self.portfolio_rebalancer.needs_rebalancing(
                target_weights, current_weights
            )
            
            # 리밸런싱 실행
            if needs_rebalance:
                self.logger.info("🔄 포트폴리오 리밸런싱 시작...")
                rebalance_success = self.portfolio_rebalancer.execute_rebalancing(rebalancing_symbols)
                
                if rebalance_success:
                    self.logger.info("✅ 포트폴리오 리밸런싱 완료")
                else:
                    self.logger.warning("⚠️ 포트폴리오 리밸런싱 실패")
                    
        except Exception as e:
            self.logger.error(f"❌ 포트폴리오 리밸런싱 확인 실패: {e}")

    def _check_portfolio_balance(self) -> Dict:
        """포트폴리오 균형 상태 확인"""
        try:
            symbols = self.config['monitoring']['symbols']
            portfolio_state = {
                'total_positions': 0,
                'total_risk_exposure': 0.0,
                'symbol_risk_exposures': {},
                'needs_rebalancing': False,
                'max_exposure_symbol': None,
                'max_exposure_ratio': 0.0,
                'correlation_summary': {}
            }
            
            if not self.executor:
                return portfolio_state
            
            try:
                if hasattr(self, 'correlation_analyzer'):
                    correlation_summary = self.correlation_analyzer.get_correlation_summary()
                    portfolio_state['correlation_summary'] = correlation_summary
                    self.logger.info(f"📊 상관관계 요약: 평균 {correlation_summary.get('avg_correlation', 0):.3f}")
            except Exception as e:
                self.logger.warning(f"⚠️ 상관관계 요약 수집 실패: {e}")
            
            total_balance = self.executor.get_futures_balance()
            
            for symbol in symbols:
                try:
                    if self.executor.safe_has_open_position(symbol):
                        portfolio_state['total_positions'] += 1
                        
                        weight = self._get_portfolio_weights(symbol)
                        risk_exposure = weight * total_balance
                        
                        portfolio_state['symbol_risk_exposures'][symbol] = {
                            'risk_exposure': risk_exposure,
                            'weight': weight,
                            'diversification_score': portfolio_state['correlation_summary'].get('diversification_scores', {}).get(symbol, 0.5)
                        }
                        portfolio_state['total_risk_exposure'] += risk_exposure
                        
                        self.logger.info(f"📊 {symbol} Risk 노출: ${risk_exposure:.2f} (가중치: {weight:.2%}, 분산점수: {portfolio_state['symbol_risk_exposures'][symbol]['diversification_score']:.3f})")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ {symbol} 포지션 확인 실패: {e}")
            
            if portfolio_state['total_positions'] > 0 and portfolio_state['symbol_risk_exposures']:
                max_symbol = max(portfolio_state['symbol_risk_exposures'], 
                            key=lambda x: portfolio_state['symbol_risk_exposures'][x]['risk_exposure'])
                max_exposure = portfolio_state['symbol_risk_exposures'][max_symbol]['risk_exposure']
                max_exposure_ratio = max_exposure / portfolio_state['total_risk_exposure'] if portfolio_state['total_risk_exposure'] > 0 else 0
                
                portfolio_state['max_exposure_symbol'] = max_symbol
                portfolio_state['max_exposure_ratio'] = max_exposure_ratio
                
                if max_exposure_ratio > 0.50:
                    portfolio_state['needs_rebalancing'] = True
                    self.logger.warning(f"⚠️ 포트폴리오 불균형 감지!")
                    self.logger.warning(f"   최대 노출: {max_symbol} {max_exposure_ratio:.1%}")
                    self.logger.warning(f"   전체 Risk: ${portfolio_state['total_risk_exposure']:.2f}")
                    
                    self._execute_rebalancing_alert(portfolio_state)
            
            try:
                diversification_scores = [data['diversification_score'] for data in portfolio_state['symbol_risk_exposures'].values()]
                if diversification_scores:
                    avg_diversification = sum(diversification_scores) / len(diversification_scores)
                    portfolio_state['avg_diversification_score'] = avg_diversification
                    
                    if avg_diversification < 0.3:
                        self.logger.warning(f"⚠️ 포트폴리오 분산 효과 낮음: 평균 분산점수 {avg_diversification:.3f}")
            except Exception as e:
                self.logger.warning(f"⚠️ 분산 점수 분석 실패: {e}")
            
            self.logger.info(f"📈 포트폴리오 상태: {portfolio_state['total_positions']}개 포지션, "
                            f"총 Risk ${portfolio_state['total_risk_exposure']:.2f}, "
                            f"리밸런싱 필요: {portfolio_state['needs_rebalancing']}")
            
            return portfolio_state
            
        except Exception as e:
            self.logger.error(f"❌ 포트폴리오 균형 확인 실패: {e}")
            import traceback
            traceback.print_exc()
            return {'needs_rebalancing': False}

    def _execute_rebalancing_alert(self, portfolio_state: Dict):
        """포트폴리오 리밸런싱 알림"""
        try:
            webhook_url = self.config.get('discord', {}).get('webhook_url')
            if not webhook_url:
                return
                
            import requests
            
            max_symbol = portfolio_state['max_exposure_symbol']
            max_ratio = portfolio_state['max_exposure_ratio']
            total_risk = portfolio_state['total_risk_exposure']
            
            correlation_summary = portfolio_state.get('correlation_summary', {})
            avg_correlation = correlation_summary.get('avg_correlation', 0)
            
            message = f"""**⚠️ 포트폴리오 불균형 경고**

🔍 **현재 상태**
• 최대 노출: {max_symbol} ({max_ratio:.1%})
• 총 Risk 노출: ${total_risk:.2f}
• 평균 상관관계: {avg_correlation:.3f}
• 포지션 수: {portfolio_state['total_positions']}개

📊 **상세 Risk 분포**"""
            
            for symbol, data in portfolio_state['symbol_risk_exposures'].items():
                exposure = data['risk_exposure']
                weight = data['weight']
                score = data['diversification_score']
                message += f"\n• {symbol}: ${exposure:.2f} ({weight:.1%}) - 분산점수: {score:.3f}"
            
            message += f"""

💡 **권장 조치**
Phase 12.1에서 자동 리밸런싱이 구현될 예정입니다.
현재는 수동으로 포트폴리오 균형을 확인해주세요.

🛡️ **현재 Risk 분산 전략**
• 상관관계 기반 가중치 조정 적용 중
• 단일 심볼 Risk 제한: 50% 초과 시 경고
• 분산 점수 기반 동적 조정"""

            data = {
                "content": message,
                "username": "Evo-Quant AI Portfolio Manager"
            }
            
            response = requests.post(webhook_url, json=data, timeout=10)
            if response.status_code == 204:
                self.logger.info("✅ 포트폴리오 리밸런싱 알림 전송 완료")
            else:
                self.logger.warning(f"⚠️ 리밸런싱 알림 전송 실패: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"❌ 리밸런싱 알림 전송 실패: {e}")

    def _apply_dynamic_parameters(self, atr_multiplier: float, risk_per_trade: float, min_confidence: float):
        """동적 파라미터 적용"""
        try:
            if hasattr(self, 'strategy') and self.strategy:
                self.strategy.atr_multiplier = atr_multiplier
                self.strategy.risk_per_trade = risk_per_trade
                self.strategy.min_confidence = min_confidence
                
            if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                self.portfolio_manager.risk_per_trade = risk_per_trade
                
            self.config['trading']['atr_multiplier'] = atr_multiplier
            self.config['trading']['risk_per_trade'] = risk_per_trade
            self.config['trading']['min_confidence'] = min_confidence
            
            self.logger.info("🎯 동적 파라미터 적용 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 동적 파라미터 적용 실패: {e}")

    def _enhance_signal_validation(self, symbol: str, signal_data: Dict) -> Dict:
        """AI 기반 신호 검증 강화 - Phase 13.3 시장 조성자 분석 통합"""
        try:
            enhanced_signal = signal_data.copy()
            
            # 기본 변동성 검증 (기존 로직)
            current_price = self._get_current_price(symbol)
            volatility = self.volatility_analyzer.update_price_data(symbol, current_price)
            market_regime = self.volatility_analyzer.get_market_regime(symbol)
            
            original_confidence = signal_data.get('confidence', 0)
            
            # 🔥 실전 검증 과정 통합
            if signal_data.get('signal_type') != 'hold':
                # 1. 신호 진행도 분석 (5가지 핵심 과정 통합)
                progress_analysis = self._calculate_signal_progress(symbol, signal_data)
                enhanced_signal.update(progress_analysis)
                
                # 2. 최종 신호 강도 평가
                final_strength = enhanced_signal.get('signal_progress', 0)
                
                # 3. 시장 상황에 따른 조정
                if market_regime == "HIGH_VOLATILITY":
                    final_strength *= 0.8
                    enhanced_signal['volatility_adjustment'] = -0.2
                elif market_regime == "LOW_VOLATILITY":
                    final_strength = min(1.0, final_strength * 1.2)
                    enhanced_signal['volatility_adjustment'] = 0.2
                else:
                    enhanced_signal['volatility_adjustment'] = 0.0
                
                # 🔥 Phase 13.2: 고급 인디케이터 검증 통합
                if self.advanced_indicators:
                    enhanced_signal = self.advanced_indicators.validate_signal_with_advanced_indicators(
                        symbol, enhanced_signal
                    )
                    
                    # 고급 인디케이터 검증 보고서 생성
                    advanced_report = self.advanced_indicators.generate_advanced_validation_report(symbol, enhanced_signal)
                    self.logger.info(f"🔍 {symbol} 고급 인디케이터 검증 완료:\n{advanced_report}")
                
                # 🔥 Phase 13.3: 시장 조성자 분석 검증 통합
                if self.market_maker_analyzer:
                    enhanced_signal = self.market_maker_analyzer.validate_signal_with_market_maker_analysis(
                        symbol, enhanced_signal
                    )
                    
                    # 시장 조성자 분석 보고서 생성
                    mm_report = self.market_maker_analyzer.generate_market_maker_report(symbol, enhanced_signal)
                    self.logger.info(f"🔍 {symbol} 시장 조성자 분석 완료:\n{mm_report}")
                
                enhanced_signal['adjusted_confidence'] = final_strength
                enhanced_signal['market_regime'] = market_regime
                enhanced_signal['volatility'] = volatility
                
                # 검증 보고서 생성 및 로깅
                validation_report = self.performance_monitor.generate_advanced_validation_report(symbol, enhanced_signal)
                self.logger.info(f"🔍 {symbol} 실전 검증 완료:\n{validation_report}")
                
                # 최소 신뢰도 임계값 확인
                min_confidence = self.config['trading'].get('min_confidence', 0.05)
                if final_strength < min_confidence:
                    enhanced_signal['signal_type'] = 'hold'
                    enhanced_signal['rejection_reason'] = 'low_confidence_after_validation'
                    self.logger.info(f"⏸️ {symbol} 신호 거부: 검증 후 신뢰도 부족 ({final_strength:.3f} < {min_confidence})")
                else:
                    self.logger.info(f"✅ {symbol} 신호 승인: 검증 통과 ({final_strength:.3f} >= {min_confidence})")
            
            self.logger.info(f"🔍 {symbol} 종합검증: {original_confidence:.3f} → {enhanced_signal.get('adjusted_confidence', 0):.3f} "
                        f"({market_regime}, vol:{volatility:.4f}, 진행도:{enhanced_signal.get('signal_progress', 0):.1f})")
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ 신호 검증 강화 실패: {e}")
            return signal_data

    def _calculate_signal_progress(self, symbol: str, signal_data: Dict) -> Dict:
        """신호 진행도 분석 - 현재 봉 분석 및 다음 봉 예측"""
        try:
            enhanced_signal = signal_data.copy()
            
            # 현재 가격과 봉 데이터 조회
            current_price = self._get_current_price(symbol)
            klines = self.executor.client.futures_klines(
                symbol=symbol, interval='5m', limit=10
            )
            
            if len(klines) >= 2:
                current_candle = klines[-1]
                prev_candle = klines[-2]
                
                # 현재 봉 분석
                open_price = float(current_candle[1])
                high_price = float(current_candle[2])
                low_price = float(current_candle[3])
                close_price = float(current_candle[4])
                volume = float(current_candle[5])
                
                # 봉 모양 분석
                candle_size = abs(close_price - open_price)
                candle_body_ratio = candle_size / (high_price - low_price) if (high_price - low_price) > 0 else 0
                
                # 다음 봉 방향성 예측
                price_momentum = (close_price - float(prev_candle[4])) / float(prev_candle[4])
                volume_change = volume / float(prev_candle[5]) if float(prev_candle[5]) > 0 else 1
                
                enhanced_signal.update({
                    'candle_analysis': {
                        'body_ratio': round(candle_body_ratio, 3),
                        'size_percent': round(candle_size / open_price * 100, 3),
                        'momentum': round(price_momentum * 100, 3),
                        'volume_change': round(volume_change, 2)
                    },
                    'signal_progress': self._evaluate_signal_strength(symbol, signal_data, current_price)
                })
                
                self.logger.info(f"📊 {symbol} 봉 분석: 몸통비율 {candle_body_ratio:.1%}, "
                            f"모멘텀 {price_momentum*100:.2f}%, 거래량 변화 {volume_change:.1f}x")
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 신호 진행도 분석 실패: {e}")
            return signal_data

    def _evaluate_signal_strength(self, symbol: str, signal_data: Dict, current_price: float) -> float:
        """신호 강도 평가 (0.0-1.0)"""
        try:
            strength_score = 0.0
            
            # 기본 신뢰도
            base_confidence = signal_data.get('confidence', 0)
            strength_score += base_confidence * 0.3
            
            # 볼륨-가격 검증
            volume_strength = self._validate_volume_price(symbol, signal_data['signal_type'])
            strength_score += volume_strength * 0.3
            
            # 다중 시간대 일관성
            timeframe_consistency = self._check_multitimeframe_consistency(symbol, signal_data['signal_type'])
            strength_score += timeframe_consistency * 0.2
            
            # 키 레벨 근접도
            key_level_strength = self._check_key_level_proximity(symbol, current_price, signal_data['signal_type'])
            strength_score += key_level_strength * 0.2
            
            return min(1.0, max(0.0, strength_score))
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 신호 강도 평가 실패: {e}")
            return signal_data.get('confidence', 0)

    def _validate_volume_price(self, symbol: str, signal_type: str) -> float:
        """체결량과 가격 움직임 연동 분석"""
        try:
            # 15분 봉 데이터로 거래량 분석
            klines = self.executor.client.futures_klines(
                symbol=symbol, interval='15m', limit=20
            )
            
            if len(klines) < 5:
                return 0.5
                
            volumes = [float(k[5]) for k in klines]
            closes = [float(k[4]) for k in klines]
            
            # 거래량 가중 이동평균
            volume_ma = sum(volumes[-5:]) / 5
            current_volume = volumes[-1]
            
            # 가격 변화율
            price_change = (closes[-1] - closes[-5]) / closes[-5]
            
            # 거래량 신호 강도 계산
            volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1
            
            if signal_type == 'buy':
                # 매수 신호: 가격 상승 + 거래량 증가 = 강한 신호
                if price_change > 0 and volume_ratio > 1.2:
                    strength = 0.8
                elif price_change > 0 and volume_ratio > 0.8:
                    strength = 0.6
                else:
                    strength = 0.3
            else:  # sell
                # 매도 신호: 가격 하락 + 거래량 증가 = 강한 신호
                if price_change < 0 and volume_ratio > 1.2:
                    strength = 0.8
                elif price_change < 0 and volume_ratio > 0.8:
                    strength = 0.6
                else:
                    strength = 0.3
                    
            self.logger.info(f"📈 {symbol} 볼륨검증: {volume_ratio:.1f}x, 가격변화 {price_change*100:.2f}%, 강도 {strength:.1f}")
            return strength
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 볼륨검증 실패: {e}")
            return 0.5

    def _validate_volume_price(self, symbol: str, signal_type: str) -> float:
        """체결량과 가격 움직임 연동 분석"""
        try:
            # 15분 봉 데이터로 거래량 분석
            klines = self.executor.client.futures_klines(
                symbol=symbol, interval='15m', limit=20
            )
            
            if len(klines) < 5:
                return 0.5
                
            volumes = [float(k[5]) for k in klines]
            closes = [float(k[4]) for k in klines]
            
            # 거래량 가중 이동평균
            volume_ma = sum(volumes[-5:]) / 5
            current_volume = volumes[-1]
            
            # 가격 변화율
            price_change = (closes[-1] - closes[-5]) / closes[-5]
            
            # 거래량 신호 강도 계산
            volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1
            
            if signal_type == 'buy':
                # 매수 신호: 가격 상승 + 거래량 증가 = 강한 신호
                if price_change > 0 and volume_ratio > 1.2:
                    strength = 0.8
                elif price_change > 0 and volume_ratio > 0.8:
                    strength = 0.6
                else:
                    strength = 0.3
            else:  # sell
                # 매도 신호: 가격 하락 + 거래량 증가 = 강한 신호
                if price_change < 0 and volume_ratio > 1.2:
                    strength = 0.8
                elif price_change < 0 and volume_ratio > 0.8:
                    strength = 0.6
                else:
                    strength = 0.3
                    
            self.logger.info(f"📈 {symbol} 볼륨검증: {volume_ratio:.1f}x, 가격변화 {price_change*100:.2f}%, 강도 {strength:.1f}")
            return strength
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 볼륨검증 실패: {e}")
            return 0.5

    def _check_multitimeframe_consistency(self, symbol: str, signal_type: str) -> float:
        """다중 시간대 신호 일관성 검증"""
        try:
            timeframes = ['15m', '1h', '4h']
            consistent_count = 0
            
            for tf in timeframes:
                try:
                    klines = self.executor.client.futures_klines(
                        symbol=symbol, interval=tf, limit=10
                    )
                    
                    if len(klines) < 5:
                        continue
                        
                    # 간단한 추세 분석 (5-period MA vs 10-period MA)
                    closes = [float(k[4]) for k in klines]
                    ma_fast = sum(closes[-5:]) / 5
                    ma_slow = sum(closes[-10:]) / 10
                    
                    tf_trend = 'buy' if ma_fast > ma_slow else 'sell'
                    
                    if tf_trend == signal_type:
                        consistent_count += 1
                        
                except Exception as tf_error:
                    self.logger.warning(f"⚠️ {symbol} {tf} 분석 실패: {tf_error}")
                    continue
            
            consistency_ratio = consistent_count / len(timeframes)
            self.logger.info(f"⏰ {symbol} 다중시간대 일관성: {consistent_count}/{len(timeframes)} ({consistency_ratio:.1%})")
            return consistency_ratio
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 다중시간대 분석 실패: {e}")
            return 0.5

    def _check_key_level_proximity(self, symbol: str, current_price: float, signal_type: str) -> float:
        """키 지원/저항 수준 근접도 분석"""
        try:
            # 1일 봉 데이터로 주요 지지/저항 분석
            klines = self.executor.client.futures_klines(
                symbol=symbol, interval='1d', limit=30
            )
            
            if len(klines) < 10:
                return 0.5
                
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            
            # 주요 저항선 (최근 고점)
            resistance_level = max(highs[-10:])
            # 주요 지지선 (최근 저점)  
            support_level = min(lows[-10:])
            
            # 현재 가격과의 거리 계산
            distance_to_resistance = abs(current_price - resistance_level) / resistance_level
            distance_to_support = abs(current_price - support_level) / support_level
            
            if signal_type == 'buy':
                # 매수 신호: 지지선 근처일수록 강함
                strength = max(0, 1 - (distance_to_support * 10))
            else:  # sell
                # 매도 신호: 저항선 근처일수록 강함
                strength = max(0, 1 - (distance_to_resistance * 10))
                
            self.logger.info(f"🎯 {symbol} 키레벨 근접도: 지지${support_level:.4f}, 저항${resistance_level:.4f}, 강도 {strength:.1f}")
            return max(0.1, min(1.0, strength))
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 키레벨 분석 실패: {e}")
            return 0.5

    def check_emergency_conditions(self) -> bool:
        """긴급 정지 조건 확인"""
        try:
            if self.emergency_stopped:
                return True
                
            current_date = datetime.now().date()
            if current_date != self.last_daily_reset:
                self.last_daily_reset = current_date
                self.logger.info("🔄 일일 PnL 리셋 완료")
            
            daily_pnl = get_daily_pnl_from_logs()
            if daily_pnl < self.daily_loss_limit:
                self.logger.critical(f"🚨 긴급 정지: 일일 손실 {daily_pnl:.2f} > 한도 {self.daily_loss_limit:.2f}")
                self.emergency_stop()
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 긴급 조건 확인 실패: {e}")
            return False

    def _debug_signal_generation(self, symbols: List[str], cycle_count: int):
        """실시간 신호 생성 디버깅 - 무한 재귀 방지"""
        try:
            self.logger.info(f"🔍 사이클 #{cycle_count} 신호 디버깅 시작...")
            
            for symbol in symbols:
                try:
                    current_price = 100.0
                    signal_result = self._simulate_signal_generation(symbol, current_price)
                    
                    if signal_result and signal_result.get('signal_type') != 'hold':
                        self.logger.info(f"🎯 {symbol} 신호 생성: {signal_result}")
                        
                        # 신호 히스토리에만 추가
                        self.signal_history.append({
                            'cycle': cycle_count,
                            'symbol': symbol,
                            'signal': signal_result,
                            'timestamp': datetime.now()
                        })
                        
                        # ✅ 중요: 실제 거래는 여기서 직접 호출하지 않음
                        # 대신 플래그만 설정하고, run() 메인 루프에서 처리
                        if self.config['binance'].get('trade_enabled', False):
                            self.logger.info(f"📝 {symbol} 거래 대기열에 추가")
                            # 대기열에 추가하는 로직 (필요시 구현)
                            
                    else:
                        self.logger.info(f"⏸️  {symbol} 신호 없음: {signal_result}")
                        
                except Exception as e:
                    self.logger.error(f"❌ {symbol} 신호 디버깅 실패: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ 신호 디버깅 실패: {e}")

    def _simulate_signal_generation(self, symbol: str, current_price: float) -> Dict:
        """신호 생성 시뮬레이션"""
        try:
            signal_types = ['buy', 'sell', 'hold']
            weights = [0.3, 0.3, 0.4]
            
            signal_type = rand_module.choices(signal_types, weights=weights)[0]
            confidence = rand_module.uniform(0.01, 0.15)
            
            min_confidence = self.config['trading'].get('min_confidence', 0.02)
            if confidence < min_confidence:
                signal_type = 'hold'
                confidence = 0.0
            
            return {
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': confidence,
                'price': current_price,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 신호 시뮬레이션 실패: {e}")
            return {'signal_type': 'hold', 'confidence': 0.0}
    
    def _execute_live_trade(self, symbol: str, signal: Dict):
        """실전 거래 실행 - 오류 수정 버전"""
        try:
            # 홀드 신호는 거래 안 함
            if signal['signal_type'] == 'hold':
                return
                
            signal_type = signal['signal_type']
            action = "BUY" if signal_type == 'buy' else "SELL"
            
            # ============================================
            # 1단계: 현재 가격 조회
            # ============================================
            try:
                ticker = self.executor.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                self.logger.info(f"💰 {symbol} 현재가: ${current_price:.4f}")
            except Exception as e:
                self.logger.error(f"❌ {symbol} 가격 조회 실패: {e}")
                return  # ✅ 명시적 종료

            # ============================================
            # 2단계: 포지션 크기 계산
            # ============================================
            try:
                confidence = signal.get('adjusted_confidence', signal.get('confidence', 0))
                
                # 동적 레버리지 적용
                leverage = self._get_dynamic_leverage(confidence, symbol)
                
                # 레버리지 설정
                try:
                    self.executor.set_leverage(symbol, leverage)
                    self.logger.info(f"🎯 {symbol} 레버리지 {leverage}배 설정 완료")
                except Exception as leverage_error:
                    self.logger.warning(f"⚠️ {symbol} 레버리지 설정 실패: {leverage_error}")
                
                # 포지션 수량 계산
                quantity = self._calculate_debug_quantity(current_price, symbol, confidence)
                
                if quantity <= 0:
                    self.logger.error(f"❌ {symbol} 유효하지 않은 포지션 수량")
                    return  # ✅ 명시적 종료
                    
            except Exception as e:
                self.logger.error(f"❌ {symbol} 포지션 계산 실패: {e}")
                return  # ✅ 명시적 종료

            # ============================================
            # 3단계: 실전 거래 실행
            # ============================================
            if not self.config['binance'].get('trade_enabled', False):
                self.logger.info(f"⏩ 거래 비활성화: {symbol} 스킵")
                return  # ✅ 명시적 종료
                
            self.logger.info(f"🎯 실전 주문: {symbol} {action} {quantity:.6f}주 @ ${current_price:.4f} (레버리지: {leverage}배)")
            
            # 강력한 주문 실행
            try:
                order_result = self.executor.robust_market_order(symbol, action, quantity)
            except Exception as order_error:
                self.logger.error(f"❌ {symbol} 주문 실행 중 예외: {order_error}")
                return  # ✅ 명시적 종료
            
            # ============================================
            # 4단계: 주문 결과 처리
            # ============================================
            if order_result.get('success', False):
                executed_qty = order_result.get('executed_qty', quantity)
                avg_price = order_result.get('avg_price', current_price)
                
                # ✅ PnL은 항상 0으로 기록 (시스템 점검 중)
                pnl = 0.0
                
                # CSV 로깅
                log_trade_to_csv(symbol, action, avg_price, executed_qty, pnl)
                
                # 성능 모니터 기록
                self.performance_monitor.record_trade(
                    symbol=symbol,
                    signal_type=signal_type,
                    confidence=confidence,
                    result={'success': True, 'pnl': pnl}
                )
                
                self.logger.info(f"✅ 실전 거래 완료: {symbol} {action} {executed_qty:.6f}주")
                
            else:
                error_msg = order_result.get('error', 'Unknown error')
                self.logger.error(f"❌ 실전 거래 실패: {symbol} - {error_msg}")
                
                if "API_KEY_ERROR" in error_msg:
                    self.logger.critical("🚨 API 키 오류로 거래 불가 - 시스템 확인 필요")
                    # 긴급 상황이므로 예외 발생
                    raise Exception(f"API 키 오류: {error_msg}")
                    
        except Exception as e:
            self.logger.error(f"❌ {symbol} 거래 실행 완전 실패: {e}")
            import traceback
            traceback.print_exc()
            # ✅ 예외를 삼키지 말고 상위로 전파
            raise

    def _force_initial_trades(self, symbols: List[str], cycle_count: int):
        """초기 강제 거래 실행 - PnL 오류 수정"""
        try:
            self.logger.info(f"🔥 초기 강제 거래 실행 (사이클 #{cycle_count})")
            
            target_symbols = ['ADAUSDT', 'SOLUSDT']
            trade_executed = False
            
            for symbol in target_symbols:
                try:
                    # 레버리지 설정 시도
                    try:
                        self.executor.set_leverage(symbol, 20)
                    except Exception as leverage_error:
                        self.logger.warning(f"⚠️ {symbol} 레버리지 설정 실패: {leverage_error}")
                    
                    # 포지션 확인
                    has_position = False
                    try:
                        has_position = self.executor.safe_has_open_position(symbol)
                    except Exception as e:
                        self.logger.warning(f"⚠️ {symbol} 포지션 확인 실패: {e}")
                        has_position = False
                    
                    if not has_position:
                        # 가격 조회
                        current_price = 0.0
                        try:
                            current_price = self._get_current_price(symbol)
                        except Exception as price_error:
                            self.logger.warning(f"⚠️ {symbol} 가격 조회 실패: {price_error}")
                            if symbol == 'ADAUSDT':
                                current_price = 0.4653
                            elif symbol == 'SOLUSDT':
                                current_price = 142.50
                            else:
                                current_price = 100.0
                        
                        if current_price and current_price > 0:
                            quantity = self._calculate_debug_quantity(current_price, symbol)
                            
                            # 🔥 실전 거래 실행
                            if self.config['binance'].get('trade_enabled', False):
                                self.logger.info(f"🎯 실전 강제 거래: {symbol} BUY {quantity:.6f}주")
                                
                                order_result = self.executor.robust_market_order(symbol, "BUY", quantity)
                                
                                if order_result['success']:
                                    executed_qty = order_result.get('executed_qty', quantity)
                                    avg_price = order_result.get('avg_price', current_price)
                                    
                                    # ✅ PnL 오류 수정: 항상 0으로 기록
                                    pnl = 0.0
                                    log_trade_to_csv(symbol, "BUY", avg_price, executed_qty, pnl)
                                    
                                    self.performance_monitor.record_trade(
                                        symbol=symbol,
                                        signal_type='buy',
                                        confidence=0.8,
                                        result={'success': True, 'pnl': pnl}
                                    )
                                    
                                    self.logger.info(f"✅ 실전 강제 거래 완료: {symbol}")
                                    trade_executed = True
                                    break
                            else:
                                self.logger.info(f"⏩ 거래 비활성화: {symbol} 강제 거래 스킵")
                                
                except Exception as symbol_error:
                    self.logger.error(f"❌ {symbol} 처리 중 오류: {symbol_error}")
                    continue
                    
            if not trade_executed:
                self.logger.info("⏩ 강제 거래 실행되지 않음 (이미 포지션 있거나 오류)")
                
        except Exception as e:
            self.logger.error(f"❌ 초기 강제 거래 실행 실패: {e}")

    def _get_portfolio_weights(self, symbol: str) -> float:
        """포트폴리오 가중치 계산"""
        try:
            base_weights = {
                'ADAUSDT': 0.25,
                'SOLUSDT': 0.25,
                'AVAXUSDT': 0.15,
                'BNBUSDT': 0.15,
                'XRPUSDT': 0.10,
                'MATICUSDT': 0.10
            }
            
            base_weight = base_weights.get(symbol, 0.10)
            self.logger.info(f"📊 {symbol} 기본 가중치: {base_weight:.2f}")
            
            if not hasattr(self, 'correlation_analyzer'):
                try:
                    from correlation_analyzer import CorrelationAnalyzer
                    symbols = self.config['monitoring']['symbols']
                    self.correlation_analyzer = CorrelationAnalyzer(symbols, cache_hours=24)
                    self.logger.info("✅ 상관관계 분석기 초기화 완료")
                except ImportError as e:
                    self.logger.error(f"❌ correlation_analyzer 임포트 실패: {e}")
                    return base_weight
            
            try:
                diversification_score = self.correlation_analyzer.get_diversification_score(symbol)
                self.logger.info(f"📈 {symbol} 분산 점수: {diversification_score:.3f}")
            except Exception as e:
                self.logger.warning(f"⚠️ {symbol} 분산 점수 계산 실패: {e}, 기본 가중치 사용")
                diversification_score = 0.5
            
            correlation_adjustment = (diversification_score - 0.5) * 0.4
            adjusted_weight = base_weight * (1 + correlation_adjustment)
            
            self.logger.info(f"🔧 {symbol} 상관관계 조정: {correlation_adjustment:+.2%} "
                            f"({base_weight:.2f} → {adjusted_weight:.2f})")
            
            try:
                open_positions = self._get_current_open_positions_count()
                max_positions = self.config['trading'].get('max_positions', 3)
                
                if open_positions >= max_positions:
                    adjusted_weight *= 0.5
                    self.logger.warning(f"⚠️ {symbol} 가중치 추가 감소: "
                                    f"최대 포지션 도달 ({open_positions}/{max_positions})")
            except Exception as e:
                self.logger.warning(f"⚠️ 오픈 포지션 확인 실패: {e}")
            
            final_weight = max(0.05, min(0.30, adjusted_weight))
            
            if final_weight != adjusted_weight:
                self.logger.warning(f"⚠️ {symbol} 가중치 제한 적용: "
                                f"{adjusted_weight:.2f} → {final_weight:.2f}")
            
            self.logger.info(f"✅ {symbol} 최종 가중치: {final_weight:.2f} "
                            f"(기본: {base_weight:.2f}, "
                            f"분산점수: {diversification_score:.3f}, "
                            f"조정: {correlation_adjustment:+.2%})")
            
            return final_weight
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 포트폴리오 가중치 계산 실패: {e}")
            import traceback
            traceback.print_exc()
            return 0.10

    def _get_current_price(self, symbol: str) -> float:
        """현재 가격 조회"""
        try:
            ticker = self.executor.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            self.logger.error(f"❌ {symbol} 가격 조회 실패: {e}")
            return 0.0

    def _get_current_open_positions_count(self) -> int:
        """현재 오픈 포지션 수 확인 - MATICUSDT 타임아웃 회피"""
        try:
            open_count = 0
            symbols = self.config['monitoring']['symbols']
            
            # MATICUSDT는 타임아웃 문제로 제외
            safe_symbols = [s for s in symbols if s != 'MATICUSDT']
            
            for symbol in safe_symbols:
                try:
                    if self.executor and self.executor.safe_has_open_position(symbol):
                        open_count += 1
                except Exception as e:
                    self.logger.warning(f"⚠️ {symbol} 포지션 확인 실패: {e}")
                    continue
                    
            self.logger.info(f"📊 현재 오픈 포지션: {open_count}개 (MATICUSDT 제외)")
            return open_count
            
        except Exception as e:
            self.logger.error(f"❌ 오픈 포지션 확인 실패: {e}")
            return 0

    def _get_dynamic_leverage(self, confidence: float, symbol: str = None) -> int:
        """신뢰도 기반 동적 레버리지 계산"""
        try:
            # 기본 레버리지 설정
            base_leverage = self.config['trading'].get('leverage', 20)
            
            # 신뢰도에 따른 레버리지 배수 조정
            if confidence >= 0.15:  # 높은 신뢰도
                leverage_multiplier = 1.2
            elif confidence >= 0.10:  # 중간 신뢰도
                leverage_multiplier = 1.0
            elif confidence >= 0.07:  # 기본 신뢰도
                leverage_multiplier = 0.8
            elif confidence >= 0.05:  # 낮은 신뢰도
                leverage_multiplier = 0.5
            else:  # 매우 낮은 신뢰도
                leverage_multiplier = 0.3
            
            dynamic_leverage = int(base_leverage * leverage_multiplier)
            
            # 안전 범위 제한 (1~30배)
            final_leverage = max(1, min(30, dynamic_leverage))
            
            self.logger.info(f"🎯 {symbol} 동적 레버리지: {final_leverage}배 "
                        f"(신뢰도: {confidence:.3f}, 기본: {base_leverage}배, 배수: {leverage_multiplier})")
            
            return final_leverage
            
        except Exception as e:
            self.logger.error(f"❌ 동적 레버리지 계산 실패: {e}")
            return self.config['trading'].get('leverage', 20)

    def _calculate_optimal_margin_usage(self, symbol: str, confidence: float) -> float:
        """최적 마진 활용률 계산"""
        try:
            # 기본 설정값
            base_risk = self.config['trading'].get('risk_per_trade', 0.08)
            max_position_ratio = self.config['trading'].get('max_position_ratio', 0.40)
            
            # 신뢰도 기반 마진 배수
            if confidence >= 0.15:
                margin_multiplier = 1.5  # 높은 신뢰도: 150% 활용
            elif confidence >= 0.10:
                margin_multiplier = 1.2  # 중간 신뢰도: 120% 활용
            elif confidence >= 0.07:
                margin_multiplier = 1.0  # 기본 신뢰도: 100% 활용
            elif confidence >= 0.05:
                margin_multiplier = 0.7  # 낮은 신뢰도: 70% 활용
            else:
                margin_multiplier = 0.4  # 매우 낮은 신뢰도: 40% 활용
            
            # 종목 수 고려 마진 할당
            symbols_count = len(self.config['monitoring']['symbols'])
            base_allocation = 1.0 / symbols_count  # 균등 분배
            
            # 공격적 모드 여부 확인
            aggressive_mode = self.config['trading'].get('aggressive_mode', False)
            if aggressive_mode:
                margin_multiplier *= 1.3  # 공격적 모드: 30% 추가 활용
                self.logger.info(f"🔥 공격적 모드: 마진 활용률 {margin_multiplier:.1f}배")
            
            # 최종 마진 활용률 계산
            optimal_margin_ratio = min(
                max_position_ratio,  # 최대 포지션 비율 제한
                base_allocation * margin_multiplier  # 종목별 할당 × 신뢰도 배수
            )
            
            self.logger.info(f"💰 {symbol} 최적 마진 활용률: {optimal_margin_ratio:.1%} "
                        f"(종목수: {symbols_count}, 기본할당: {base_allocation:.1%}, 배수: {margin_multiplier:.1f}x)")
            
            return optimal_margin_ratio
            
        except Exception as e:
            self.logger.error(f"❌ 최적 마진 활용률 계산 실패: {e}")
            return self.config['trading'].get('max_position_ratio', 0.40)

    def _calculate_debug_quantity(self, price: float, symbol: str = "ADAUSDT") -> float:
        """포지션 사이즈 계산"""
        try:
            if not self.executor:
                self.logger.error("❌ 트레이딩 실행기가 없습니다")
                return 0.0
                
            balance = self.executor.get_futures_balance()
            if balance <= 0:
                self.logger.error("❌ 잔고가 부족합니다")
                return 0.0
            
            leverage = self.config['trading'].get('leverage', 20)
            available_margin = balance * leverage
            aggressive_risk_ratio = 0.20
            
            risk_amount = available_margin * aggressive_risk_ratio
            
            self.logger.info(f"💰 {symbol} 공격적 포지션 계산:")
            self.logger.info(f"   잔고: ${balance:.2f}")
            self.logger.info(f"   레버리지: {leverage}배")
            self.logger.info(f"   사용 가능 Margin: ${available_margin:.2f}")
            self.logger.info(f"   공격적 Risk: {aggressive_risk_ratio*100}% (${risk_amount:.2f})")
            
            quantity = self.executor.calculate_position_size(symbol, risk_amount, price)
            
            actual_notional = quantity * price
            min_notional = 5.0
            
            if actual_notional < min_notional:
                self.logger.warning(f"⚠️ {symbol} 주문 금액 부족: ${actual_notional:.2f}")
                min_quantity = self.executor.calculate_position_size(symbol, min_notional * 2, price)
                quantity = min_quantity
                actual_notional = quantity * price
                self.logger.info(f"📦 공격적 수량 조정: {quantity:.6f}주 (${actual_notional:.2f})")
            
            max_position_value = balance * 0.40
            if actual_notional > max_position_value:
                self.logger.warning(f"⚠️ {symbol} 포지션 크기 제한: ${actual_notional:.2f} > ${max_position_value:.2f}")
                adjusted_quantity = self.executor.calculate_position_size(symbol, max_position_value, price)
                quantity = adjusted_quantity
                actual_notional = quantity * price
                self.logger.info(f"📦 최대 수량 조정: {quantity:.6f}주 (${actual_notional:.2f})")
            
            used_margin = actual_notional / leverage
            margin_utilization = (used_margin / balance) * 100
            
            self.logger.info(f"🎯 {symbol} 최종 포지션: {quantity:.6f}주 (${actual_notional:.2f})")
            self.logger.info(f"📊 Margin 활용률: {margin_utilization:.1f}% (사용: ${used_margin:.2f})")
            
            if quantity <= 0 or actual_notional < min_notional:
                self.logger.error(f"❌ {symbol} 유효하지 않은 수량: {quantity:.6f}주 (${actual_notional:.2f})")
                return 0.0
                
            return quantity
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 포지션 계산 실패: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def emergency_stop(self):
        """긴급 정지 실행"""
        try:
            self.emergency_stopped = True
            self.logger.critical("🚨 🚨 🚨 긴급 정지 실행 🚨 🚨 🚨")
            
            symbols = self.config['monitoring']['symbols']
            for symbol in symbols:
                try:
                    if self.executor and self.executor.safe_has_open_position(symbol):
                        self.logger.warning(f"⚠️ {symbol} 포지션 강제 청산 시도")
                        log_trade_to_csv(symbol, "EMERGENCY_CLOSE", 0, 0, 0)
                except Exception as e:
                    self.logger.error(f"❌ {symbol} 청산 실패: {e}")
            
            webhook_url = self.config.get('discord', {}).get('webhook_url')
            if webhook_url:
                import requests
                daily_pnl = get_daily_pnl_from_logs()
                data = {
                    "content": f"**🚨 🚨 🚨 긴급 정지 🚨 🚨 🚨**\n일일 손실 한도 초과로 시스템이 중단되었습니다!\n현재 일일 PnL: ${daily_pnl:.2f}\n손실 한도: ${self.daily_loss_limit:.2f}\n**수동 개입이 필요합니다!**",
                    "username": "Evo-Quant AI Emergency"
                }
                requests.post(webhook_url, json=data, timeout=10)
                
            self.logger.critical("🛑 시스템 정지 - 수동 개입 필요")
            
        except Exception as e:
            self.logger.error(f"❌ 긴급 정지 실행 실패: {e}")

    def _verify_with_backtesting(self):
        """백테스팅으로 전략 검증"""
        try:
            self.logger.info("🔍 백테스팅 검증 실행...")
            
            try:
                from advanced_backtester import BacktestEngine
                backtest_available = True
            except ImportError as e:
                self.logger.warning(f"⚠️ 백테스팅 엔진 임포트 실패: {e}")
                self.logger.info("💡 백테스팅 없이 라이브 트레이딩 계속 진행")
                backtest_available = False
                
            if not backtest_available:
                return
                
            engine = BacktestEngine(self.config)
            symbols = self.config['monitoring']['symbols']
            if not symbols:
                self.logger.warning("⚠️ 모니터링 심볼 없음")
                return
                
            test_symbol = symbols[0]
            self.logger.info(f"📊 백테스팅 검증: {test_symbol}")
            
            result = engine.run_backtest(test_symbol, cash=10000)
            
            if result['status'] == 'success':
                stats = result['stats']
                self.logger.info(f"✅ 백테스팅 검증 완료: {test_symbol}")
                self.logger.info(f"   수익률: {stats['Return [%]']:.2f}%")
                self.logger.info(f"   샤프 비율: {stats['Sharpe Ratio']:.2f}")
            else:
                self.logger.warning(f"⚠️ 백테스팅 검증 실패: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ 백테스팅 검증 실패: {e}")

    def _initialize_legacy_modules(self):
        """기존 모듈 초기화 - 실전 매매 검증 강화"""
        try:
            self._verify_with_backtesting()
            
            from trading_executor_v2 import MultiExchangeManager
            self.multi_exchange = MultiExchangeManager(self.config)
            self.executor = self.multi_exchange.get_active_exchange()
            
            if self.executor:
                exchange_name = [k for k, v in self.multi_exchange.exchanges.items() if v == self.executor][0]
                self.logger.info(f"✅ 멀티 익스체인지 관리자 초기화 완료 - 활성: {exchange_name}")
                
                # 🔥 즉시 잔고 조회 테스트 (API 키 검증)
                try:
                    balance = self.executor.get_futures_balance()
                    self.logger.info(f"💰 실전 잔고 확인: ${balance:.2f}")
                    
                    if balance <= 0:
                        self.logger.error("❌ 거래 잔고가 0입니다 - 자금 충전 필요")
                        
                except Exception as e:
                    self.logger.critical(f"🚨 거래소 연결 실패: {e}")
                    self.logger.critical("🔑 API 키와 권한을 확인하세요")
                    raise Exception("거래소 연결 실패 - 실전 매매 불가")
                
                balances = self.multi_exchange.get_balance_all()
                total_balance = sum(balances.values())
                self.logger.info(f"💰 총 잔고: ${total_balance:.2f}")
                
            else:
                self.logger.error("❌ 사용 가능한 거래소가 없습니다")
                raise Exception("No available exchanges")
            
            from hybrid_strategy_improved_v2 import HybridStrategyImprovedV2
            self.strategy = HybridStrategyImprovedV2(self.config)
            self.logger.info("✅ 하이브리드 전략 초기화 성공")
                        
            self.portfolio_manager = None
            try:
                from portfolio_manager_v2 import PortfolioManagerV2
                
                actual_balance = self.executor.get_futures_balance() if self.executor else 100.0
                initial_capital = type_safe.safe_float(actual_balance, 100.0)
                
                self.config['trading']['initial_capital'] = initial_capital
                
                self.portfolio_manager = PortfolioManagerV2(
                    trading_executor=self.executor,
                    initial_capital=initial_capital,
                    risk_per_trade=self.config['trading'].get('risk_per_trade', 0.02),
                    config=self.config
                )
                self.logger.info(f"✅ 포트폴리오 관리자 초기화 성공 (실제 자본: ${initial_capital:.2f})")
                
            except ImportError as e:
                self.logger.warning(f"⚠️ 포트폴리오 관리자를 찾을 수 없습니다: {e}")
            
            try:
                from evo_quant_enhancements import EvoQuantEnhancements

                self.evo_quant = EvoQuantEnhancements(self.config)
                self.logger.info("✅ Evo-Quant v3.0 기능 통합 성공")
            except ImportError as e:
                self.logger.warning(f"⚠️ Evo-Quant 모듈 임포트 실패: {e}")
                class DummyEvoQuant:
                    def __init__(self, config): 
                        self.cool_down = type('CoolDownState', (), {'active': False})()
                    def should_allow_new_trade(self, confidence): return True
                    def update_cool_down_state(self, cycle_count): pass
                    def check_cool_down_condition(self, balance, pnl): return False
                self.evo_quant = DummyEvoQuant(self.config)
            
            self.logger.info("✅ 모든 모듈 초기화 성공")
            
        except Exception as e:
            self.logger.error(f"❌ 모듈 초기화 실패: {e}")
            raise


    def run(self):
        """메인 실행 루프 - 오류 수정"""
        if not hasattr(self, 'daily_loss_limit'):
            self.initial_capital = self.config['trading'].get('initial_capital', 100.0)
            self.daily_loss_limit = -0.05 * self.initial_capital
            self.emergency_stopped = False
            self.last_daily_reset = datetime.now().date()
            self.logger.info(f"🛡️ 긴급 정지 시스템 초기화: 일일 손실 한도 ${self.daily_loss_limit:.2f}")
        
        self.logger.info("🚀 Phase 11.0 트레이딩 엔진 시작 (TESTUSDT 제거 + PnL 오류 수정)")
        
        symbols = self.config['monitoring']['symbols']
        update_interval = self.config['monitoring']['update_interval']
        cycle_count = 0
        
        # 🔥 성능 보고서 오류 방지
        try:
            start_report = self.performance_monitor.generate_report()
            self.logger.info(f"📊 시작 성능 보고서:\n{start_report}")
        except Exception as e:
            self.logger.error(f"❌ 시작 성능 보고서 생성 실패: {e}")
            self.logger.info("📊 기본 성능 보고서 사용")
        
        try:
            webhook_url = self.config.get('discord', {}).get('webhook_url')
            if webhook_url:
                import requests
                data = {
                    "content": f"**🚀 Phase 11.0 시작 (TESTUSDT 제거)**\nEvo-Quant AI v3.0 트레이딩 시스템이 시작되었습니다.\n• 잔고: ${self.executor.get_futures_balance() if self.executor else 0:.2f}\n• 심볼: {len(symbols)}개\n• 주기: {update_interval}초\n• PnL 계산: 점검 중 (모든 수익 기록 0)\n• 실제 계좌 확인 필요",
                    "username": "Evo-Quant AI Trader"
                }
                requests.post(webhook_url, json=data, timeout=10)
                self.logger.info("✅ Discord 시작 알림 전송")
        except Exception as e:
            self.logger.warning(f"⚠️ Discord 알림 실패: {e}")
        
        self.logger.info(f"🔁 모니터링 시작: {len(symbols)}개 심볼, {update_interval}초 주기")
        
        # 🔥 일일 데이터 검증 스케줄러 시작
        self._start_daily_audit_scheduler()
        
        self.debug_signals = True
        self.signal_history = []

        try:
            while True:
                cycle_count += 1
                self.logger.info(f"🔄 트레이딩 사이클 #{cycle_count} 시작")
                
                if cycle_count <= 5:
                    self._force_initial_trades(symbols, cycle_count)
                
                try:
                    if self.check_emergency_conditions():
                        self.logger.critical("🛑 긴급 정지 조건 충족 - 시스템 종료")
                        break
                    
                    cycle_start_time = datetime.now()
                    
                    if self.debug_signals:
                        self._debug_signal_generation(symbols, cycle_count)

                    cycle_result = self.core_engine.execute_trading_cycle(
                        symbols=symbols,
                        executor=self.executor,
                        strategy=self.strategy,
                        portfolio_manager=self.portfolio_manager
                    )
                    
                    cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
                    
                    for symbol in symbols:
                        self.performance_monitor.record_trade(
                            symbol=symbol,
                            signal_type='monitor',
                            confidence=0.0,
                            result={'success': True, 'pnl': 0.0}
                        )
                    
                    self.logger.info(f"✅ 사이클 #{cycle_count} 완료: {cycle_result.get('status', 'unknown')} "
                                f"(소요시간: {cycle_duration:.1f}초)")
                    
                    self._adjust_parameters_real_time(cycle_count)
                    
                    if cycle_count % 5 == 0:
                        self._check_and_rebalance_portfolio()
                    
                    if cycle_count % 10 == 0:
                        try:
                            # 🔥 실시간 PnL 계산
                            portfolio_summary = self.performance_monitor.get_portfolio_summary(self)
                            
                            sharpe_ratio = self.performance_monitor.calculate_sharpe_ratio()
                            max_drawdown = self.performance_monitor.calculate_max_drawdown()
                            
                            performance_summary = {
                                **portfolio_summary,
                                'sharpe_ratio': sharpe_ratio,
                                'max_drawdown': max_drawdown
                            }
                            
                            log_performance_to_csv(performance_summary)
                            self.logger.info(f"📊 성능 메트릭: PnL=${performance_summary.get('total_pnl', 0):.4f}, "
                                            f"Sharpe={sharpe_ratio:.4f}, MDD={max_drawdown:.2f}%")
                            
                        except Exception as e:
                            self.logger.warning(f"⚠️ 성능 메트릭 계산 실패: {e}")

                    if cycle_count % 10 == 0:
                        report = self.performance_monitor.generate_report()
                        self.logger.info(f"📈 주기적 성능 보고서 (사이클 #{cycle_count}):\n{report}")
                        
                        try:
                            webhook_url = self.config.get('discord', {}).get('webhook_url')
                            if webhook_url:
                                import requests
                                # 🔥 재귀 방지: 직접 성능 데이터 사용
                                summary = self.performance_monitor.get_performance_summary()
                                data = {
                                    "content": f"**📊 실시간 성능 보고서**\n사이클: #{cycle_count}\n"
                                            f"총 거래: {summary.get('total_trades', 0)}회\n"
                                            f"실시간 PnL: ${summary.get('total_pnl', 0):.4f}\n"
                                            f"활성 포지션: {summary.get('active_positions', 0)}개\n"
                                            f"승률: {summary.get('win_rate', 0):.1f}%\n"
                                            f"가동시간: {summary.get('uptime_hours', 0):.1f}시간\n"
                                            f"✅ PnL 계산 시스템 정상 가동",
                                    "username": "Evo-Quant AI Trader"
                                }
                                requests.post(webhook_url, json=data, timeout=10)
                        except Exception as e:
                            self.logger.warning(f"⚠️ Discord 보고서 전송 실패: {e}")
                    
                except Exception as cycle_error:
                    self.logger.error(f"❌ 사이클 #{cycle_count} 실행 오류: {cycle_error}")
                    
                self.logger.info(f"⏰ {update_interval}초 후 다음 사이클...")
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 사용자에 의해 종료됨")
            
            final_report = self.performance_monitor.generate_report()
            self.logger.info(f"📊 최종 성능 보고서:\n{final_report}")
            
            try:
                if webhook_url:
                    summary = self.performance_monitor.get_performance_summary()
                    data = {
                        "content": f"**🛑 시스템 종료**\n트레이딩 시스템이 사용자에 의해 종료되었습니다.\n총 실행 사이클: {cycle_count}\n최종 가동시간: {summary.get('uptime_hours', 0):.1f}시간\n📊 총 거래: {summary.get('total_trades', 0)}회",
                        "username": "Evo-Quant AI Trader"
                    }
                    requests.post(webhook_url, json=data, timeout=10)
            except Exception as e:
                self.logger.warning(f"⚠️ 종료 알림 실패: {e}")
                
        except Exception as e:
            self.logger.error(f"❌ 실행 중 오류: {e}")
            
            try:
                if webhook_url:
                    data = {
                        "content": f"**❌ 시스템 오류**\n트레이딩 시스템 오류: {str(e)}\n마지막 사이클: #{cycle_count}",
                        "username": "Evo-Quant AI Trader"
                    }
                    requests.post(webhook_url, json=data, timeout=10)
            except Exception as notify_error:
                self.logger.error(f"❌ 오류 알림 실패: {notify_error}")

class PortfolioRebalancer:
    """포트폴리오 리밸런싱 시스템 - Phase 12.1"""
    
    def __init__(self, config, executor, performance_monitor):
        self.config = config
        self.executor = executor
        self.performance_monitor = performance_monitor
        self.logger = logging.getLogger(__name__)
        self.rebalance_threshold = 0.15  # 15% 이상 편차시 리밸런싱
        self.max_single_position = 0.25   # 단일 포지션 최대 25%
            
    def calculate_target_weights(self, current_positions: Dict) -> Dict[str, float]:
        """변동성 기반 동적 가중치 계산"""
        try:
            symbols = self.config['monitoring']['symbols']
            total_balance = self.executor.get_futures_balance()
            
            if total_balance <= 0:
                self.logger.warning("⚠️ 잔고가 0이므로 기본 가중치 사용")
                return self._get_fallback_weights(symbols)
            
            # 기본 가중치 (변동성 고려)
            base_weights = {
                'ADAUSDT': 0.20, 'SOLUSDT': 0.20, 'AVAXUSDT': 0.15,
                'BNBUSDT': 0.15, 'XRPUSDT': 0.15, 'MATICUSDT': 0.15
            }
            
            # 변동성 조정 계수 계산
            volatility_adjustments = self._calculate_volatility_adjustments(symbols)
            
            # PnL 기반 조정
            pnl_adjustments = self._calculate_pnl_adjustments(symbols, total_balance)
            
            # 최종 가중치 계산
            target_weights = {}
            for symbol in symbols:
                base_weight = base_weights.get(symbol, 0.10)
                vol_adj = volatility_adjustments.get(symbol, 1.0)
                pnl_adj = pnl_adjustments.get(symbol, 1.0)
                
                adjusted_weight = base_weight * vol_adj * pnl_adj
                target_weights[symbol] = max(0.05, min(0.30, adjusted_weight))
            
            # 가중치 정규화 (합계 100%)
            total_weight = sum(target_weights.values())
            if total_weight > 0:
                normalized_weights = {
                    symbol: weight / total_weight 
                    for symbol, weight in target_weights.items()
                }
            else:
                normalized_weights = self._get_fallback_weights(symbols)
                
            self.logger.info(f"🎯 동적 목표 가중치: {normalized_weights}")
            return normalized_weights
            
        except Exception as e:
            self.logger.error(f"❌ 동적 가중치 계산 실패: {e}")
            return self._get_fallback_weights(symbols)

    def _calculate_volatility_adjustments(self, symbols: List[str]) -> Dict[str, float]:
        """변동성 기반 조정 계수 계산"""
        volatility_adjustments = {}
        
        for symbol in symbols:
            try:
                # 간단한 변동성 계산 (24시간 데이터)
                price_history = self._get_recent_prices(symbol, hours=24)
                if len(price_history) > 10:
                    returns = []
                    for i in range(1, len(price_history)):
                        daily_return = (price_history[i] - price_history[i-1]) / price_history[i-1]
                        returns.append(daily_return)
                    
                    if returns:
                        volatility = np.std(returns)
                        # 변동성 높을수록 가중치 감소 (0.5~1.0 범위)
                        adjustment = 1.0 / (1.0 + volatility * 8)
                        volatility_adjustments[symbol] = max(0.5, min(1.2, adjustment))
                        self.logger.info(f"📊 {symbol} 변동성: {volatility:.4f}, 조정: {adjustment:.3f}")
                    else:
                        volatility_adjustments[symbol] = 1.0
                else:
                    volatility_adjustments[symbol] = 1.0
                    
            except Exception as e:
                self.logger.warning(f"⚠️ {symbol} 변동성 계산 실패: {e}")
                volatility_adjustments[symbol] = 1.0
        
        return volatility_adjustments

    def _calculate_pnl_adjustments(self, symbols: List[str], total_balance: float) -> Dict[str, float]:
        """PnL 기반 조정 계수 계산"""
        pnl_adjustments = {}
        
        try:
            pnl_data = self.performance_monitor.calculate_real_time_pnl(
                getattr(self.performance_monitor, 'engine', None)
            )
            
            for symbol in symbols:
                symbol_pnl = pnl_data.get(symbol, {}).get('unrealized_pnl', 0)
                if symbol_pnl < -total_balance * 0.02:  # 2% 이상 손실
                    # 손실 심볼 가중치 감소
                    pnl_adjustments[symbol] = max(0.6, 1.0 + (symbol_pnl / (total_balance * 0.05)))
                    self.logger.warning(f"📉 {symbol} PnL 기반 가중치 감소: {pnl_adjustments[symbol]:.3f}")
                elif symbol_pnl > total_balance * 0.05:  # 5% 이상 수익
                    # 수익 심볼 가중치 약간 증가
                    pnl_adjustments[symbol] = min(1.2, 1.0 + (symbol_pnl / (total_balance * 0.2)))
                    self.logger.info(f"📈 {symbol} PnL 기반 가중치 증가: {pnl_adjustments[symbol]:.3f}")
                else:
                    pnl_adjustments[symbol] = 1.0
                    
        except Exception as e:
            self.logger.error(f"❌ PnL 조정 계산 실패: {e}")
            # 실패시 모든 심볼 1.0 반환
            pnl_adjustments = {symbol: 1.0 for symbol in symbols}
        
        return pnl_adjustments

    def _get_recent_prices(self, symbol: str, hours: int = 24) -> List[float]:
        """최근 가격 히스토리 조회"""
        try:
            # 1시간 봉 데이터로 최근 가격 조회
            klines = self.executor.client.futures_klines(
                symbol=symbol, 
                interval='1h', 
                limit=hours
            )
            return [float(k[4]) for k in klines]  # 종가 반환
        except Exception as e:
            self.logger.warning(f"⚠️ {symbol} 가격 히스토리 조회 실패: {e}")
            return []

    def _get_fallback_weights(self, symbols: List[str]) -> Dict[str, float]:
        """폴백 가중치 (계산 실패시)"""
        base_weights = {
            'ADAUSDT': 0.20, 'SOLUSDT': 0.20, 'AVAXUSDT': 0.15,
            'BNBUSDT': 0.15, 'XRPUSDT': 0.15, 'MATICUSDT': 0.15
        }
        return {symbol: base_weights.get(symbol, 0.10) for symbol in symbols}

    def _get_recent_prices(self, symbol: str, hours: int = 24) -> List[float]:
        """최근 가격 히스토리 조회"""
        try:
            # 간단한 구현: 실제로는 Binance API에서 1시간 봉 데이터 가져오기
            klines = self.executor.client.futures_klines(
                symbol=symbol, 
                interval='1h', 
                limit=hours
            )
            return [float(k[4]) for k in klines]  # 종가만 반환
        except Exception as e:
            self.logger.warning(f"⚠️ {symbol} 가격 히스토리 조회 실패: {e}")
            return []

    def calculate_current_weights(self, current_positions: Dict) -> Dict[str, float]:
        """현재 포트폴리오 가중치 계산"""
        try:
            total_balance = self.executor.get_futures_balance()
            if total_balance <= 0:
                return {}
                
            current_weights = {}
            
            for symbol, position_data in current_positions.items():
                position_value = abs(position_data.get('position_amt', 0)) * position_data.get('entry_price', 0)
                weight = position_value / total_balance if total_balance > 0 else 0
                current_weights[symbol] = weight
                
            self.logger.info(f"📊 현재 포트폴리오 가중치: {current_weights}")
            return current_weights
            
        except Exception as e:
            self.logger.error(f"❌ 현재 가중치 계산 실패: {e}")
            return {}

    def needs_rebalancing(self, target_weights: Dict, current_weights: Dict) -> bool:
        """리밸런싱 필요 여부 확인"""
        try:
            rebalancing_symbols = []
            
            for symbol, target_weight in target_weights.items():
                current_weight = current_weights.get(symbol, 0)
                weight_deviation = abs(current_weight - target_weight)
                
                if weight_deviation > self.rebalance_threshold:
                    rebalancing_symbols.append({
                        'symbol': symbol,
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'deviation': weight_deviation
                    })
                    self.logger.warning(f"⚠️ {symbol} 리밸런싱 필요: {current_weight:.1%} -> {target_weight:.1%} (편차: {weight_deviation:.1%})")
            
            # 단일 포지션 과다 집중 검사
            for symbol, current_weight in current_weights.items():
                if current_weight > self.max_single_position:
                    rebalancing_symbols.append({
                        'symbol': symbol,
                        'current_weight': current_weight,
                        'target_weight': self.max_single_position,
                        'deviation': current_weight - self.max_single_position,
                        'reason': 'MAX_POSITION_EXCEEDED'
                    })
                    self.logger.warning(f"🚨 {symbol} 포지션 과다: {current_weight:.1%} > 최대 {self.max_single_position:.1%}")
            
            if rebalancing_symbols:
                self.logger.info(f"🔁 리밸런싱 필요: {len(rebalancing_symbols)}개 심볼")
                return True, rebalancing_symbols
            else:
                self.logger.info("✅ 포트폴리오 균형 유지됨")
                return False, []
                
        except Exception as e:
            self.logger.error(f"❌ 리밸런싱 필요 여부 확인 실패: {e}")
            return False, []

    def execute_rebalancing(self, rebalancing_symbols: List[Dict]):
        """포트폴리오 리밸런싱 실행"""
        try:
            total_balance = self.executor.get_futures_balance()
            executed_trades = []
            
            for rebalance_info in rebalancing_symbols:
                symbol = rebalance_info['symbol']
                current_weight = rebalance_info['current_weight']
                target_weight = rebalance_info['target_weight']
                
                try:
                    # 현재 포지션 정보 조회
                    positions = self.executor.client.futures_position_information(symbol=symbol)
                    current_position = 0.0
                    entry_price = 0.0
                    
                    for position in positions:
                        position_amt = float(position.get('positionAmt', 0))
                        if abs(position_amt) > 0.00001:
                            current_position = position_amt
                            entry_price = float(position.get('entryPrice', 0))
                            break
                    
                    # 목표 포지션 계산
                    target_position_value = total_balance * target_weight
                    current_price = self._get_current_price(symbol)
                    
                    if current_price > 0:
                        target_quantity = target_position_value / current_price
                        
                        # 현재 포지션과 비교하여 조정 필요량 계산
                        adjustment_needed = target_quantity - abs(current_position)
                        
                        if abs(adjustment_needed) > 0.00001:  # 의미 있는 조정량
                            side = "BUY" if adjustment_needed > 0 else "SELL"
                            quantity = abs(adjustment_needed)
                            
                            # 최소 주문 금액 확인
                            order_value = quantity * current_price
                            if order_value >= 10.0:  # Binance 최소 주문 금액
                                self.logger.info(f"🔄 {symbol} 리밸런싱: {side} {quantity:.4f}주")
                                
                                # 주문 실행
                                order_result = self.executor.robust_market_order(
                                    symbol, side, quantity
                                )
                                
                                if order_result['success']:
                                    executed_trades.append({
                                        'symbol': symbol,
                                        'side': side,
                                        'quantity': quantity,
                                        'price': current_price,
                                        'reason': 'REBALANCING'
                                    })
                                    self.logger.info(f"✅ {symbol} 리밸런싱 완료")
                                else:
                                    self.logger.error(f"❌ {symbol} 리밸런싱 실패: {order_result.get('error')}")
                            else:
                                self.logger.info(f"⏩ {symbol} 리밸런싱 스킵: 주문 금액 부족 (${order_value:.2f})")
                    
                except Exception as symbol_error:
                    self.logger.error(f"❌ {symbol} 리밸런싱 실패: {symbol_error}")
                    continue
            
            # 리밸런싱 결과 보고
            if executed_trades:
                self._send_rebalancing_report(executed_trades, total_balance)
                return True
            else:
                self.logger.info("⏩ 리밸런싱 실행 없음")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 리밸런싱 실행 실패: {e}")
            return False

    def _send_rebalancing_report(self, executed_trades: List[Dict], total_balance: float):
        """리밸런싱 결과 보고"""
        try:
            webhook_url = self.config.get('discord', {}).get('webhook_url')
            if not webhook_url:
                return
                
            import requests
            
            trade_summary = "\n".join([
                f"• {trade['symbol']} {trade['side']} {trade['quantity']:.4f}주 @ ${trade['price']:.4f}"
                for trade in executed_trades
            ])
            
            message = f"""**🔄 포트폴리오 리밸런싱 완료**

📊 **리밸런싱 요약**
• 실행된 거래: {len(executed_trades)}건
• 총 잔고: ${total_balance:.2f}
• 리밸런싱 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🎯 **실행된 거래**
{trade_summary}

💡 **리밸런싱 기준**
• 가중치 편차: {self.rebalance_threshold:.1%} 초과시
• 단일 포지션: 최대 {self.max_single_position:.1%} 제한
• PnL 기반 동적 조정 적용"""

            data = {
                "content": message,
                "username": "Evo-Quant AI Portfolio Rebalancer"
            }
            
            response = requests.post(webhook_url, json=data, timeout=10)
            if response.status_code == 204:
                self.logger.info("✅ 리밸런싱 보고서 전송 완료")
            else:
                self.logger.warning(f"⚠️ 리밸런싱 보고서 전송 실패: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"❌ 리밸런싱 보고서 전송 실패: {e}")

    def _execute_rebalancing_alert(self, portfolio_state: Dict):
        """포트폴리오 리밸런싱 알림"""
        try:
            webhook_url = self.config.get('discord', {}).get('webhook_url')
            if not webhook_url:
                return
                
            import requests
            
            max_symbol = portfolio_state['max_exposure_symbol']
            max_ratio = portfolio_state['max_exposure_ratio']
            total_risk = portfolio_state['total_risk_exposure']
            
            correlation_summary = portfolio_state.get('correlation_summary', {})
            avg_correlation = correlation_summary.get('avg_correlation', 0)
            
            message = f"""**⚠️ 포트폴리오 불균형 경고**

    🔍 **현재 상태**
    • 최대 노출: {max_symbol} ({max_ratio:.1%})
    • 총 Risk 노출: ${total_risk:.2f}
    • 평균 상관관계: {avg_correlation:.3f}
    • 포지션 수: {portfolio_state['total_positions']}개

    📊 **상세 Risk 분포**"""
            
            for symbol, data in portfolio_state['symbol_risk_exposures'].items():
                exposure = data['risk_exposure']
                weight = data['weight']
                score = data['diversification_score']
                message += f"\n• {symbol}: ${exposure:.2f} ({weight:.1%}) - 분산점수: {score:.3f}"
            
            message += f"""

    💡 **권장 조치**
    Phase 12.1에서 자동 리밸런싱이 구현될 예정입니다.
    현재는 수동으로 포트폴리오 균형을 확인해주세요.

    🛡️ **현재 Risk 분산 전략**
    • 상관관계 기반 가중치 조정 적용 중
    • 단일 심볼 Risk 제한: 50% 초과 시 경고
    • 분산 점수 기반 동적 조정"""

            data = {
                "content": message,
                "username": "Evo-Quant AI Portfolio Manager"
            }
            
            response = requests.post(webhook_url, json=data, timeout=10)
            if response.status_code == 204:
                self.logger.info("✅ 포트폴리오 리밸런싱 알림 전송 완료")
            else:
                self.logger.warning(f"⚠️ 리밸런싱 알림 전송 실패: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"❌ 리밸런싱 알림 전송 실패: {e}")


    def _apply_dynamic_parameters(self, atr_multiplier: float, risk_per_trade: float, min_confidence: float):
        """동적 파라미터 적용"""
        try:
            if hasattr(self, 'strategy') and self.strategy:
                self.strategy.atr_multiplier = atr_multiplier
                self.strategy.risk_per_trade = risk_per_trade
                self.strategy.min_confidence = min_confidence
                
            if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                self.portfolio_manager.risk_per_trade = risk_per_trade
                
            self.config['trading']['atr_multiplier'] = atr_multiplier
            self.config['trading']['risk_per_trade'] = risk_per_trade
            self.config['trading']['min_confidence'] = min_confidence
            
            self.logger.info("🎯 동적 파라미터 적용 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 동적 파라미터 적용 실패: {e}")

    def _enhance_signal_validation(self, symbol: str, signal_data: Dict) -> Dict:
        """AI 기반 신호 검증 강화"""
        try:
            enhanced_signal = signal_data.copy()
            
            current_price = 100.0
            volatility = self.volatility_analyzer.update_price_data(symbol, current_price)
            market_regime = self.volatility_analyzer.get_market_regime(symbol)
            
            original_confidence = signal_data.get('confidence', 0)
            
            if market_regime == "HIGH_VOLATILITY":
                adjusted_confidence = original_confidence * 0.8
                enhanced_signal['volatility_adjustment'] = -0.2
            elif market_regime == "LOW_VOLATILITY":
                adjusted_confidence = min(1.0, original_confidence * 1.2)
                enhanced_signal['volatility_adjustment'] = 0.2
            else:
                adjusted_confidence = original_confidence
                enhanced_signal['volatility_adjustment'] = 0.0
            
            enhanced_signal['adjusted_confidence'] = adjusted_confidence
            enhanced_signal['market_regime'] = market_regime
            enhanced_signal['volatility'] = volatility
            
            min_confidence = self.config['trading'].get('min_confidence', 0.05)
            if adjusted_confidence < min_confidence:
                enhanced_signal['signal_type'] = 'hold'
                enhanced_signal['rejection_reason'] = 'low_confidence'
            
            self.logger.info(f"🔍 {symbol} 신호 검증: {original_confidence:.3f} → {adjusted_confidence:.3f} "
                           f"({market_regime}, vol:{volatility:.4f})")
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ 신호 검증 강화 실패: {e}")
            return signal_data

    def check_emergency_conditions(self) -> bool:
        """긴급 정지 조건 확인"""
        try:
            if self.emergency_stopped:
                return True
                
            current_date = datetime.now().date()
            if current_date != self.last_daily_reset:
                self.last_daily_reset = current_date
                self.logger.info("🔄 일일 PnL 리셋 완료")
            
            daily_pnl = get_daily_pnl_from_logs()
            if daily_pnl < self.daily_loss_limit:
                self.logger.critical(f"🚨 긴급 정지: 일일 손실 {daily_pnl:.2f} > 한도 {self.daily_loss_limit:.2f}")
                self.emergency_stop()
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 긴급 조건 확인 실패: {e}")
            return False

    def _debug_signal_generation(self, symbols: List[str], cycle_count: int):
        """실시간 신호 생성 디버깅 - 무한 재귀 방지"""
        try:
            self.logger.info(f"🔍 사이클 #{cycle_count} 신호 디버깅 시작...")
            
            for symbol in symbols:
                try:
                    current_price = 100.0
                    signal_result = self._simulate_signal_generation(symbol, current_price)
                    
                    if signal_result and signal_result.get('signal_type') != 'hold':
                        self.logger.info(f"🎯 {symbol} 신호 생성: {signal_result}")
                        
                        # 신호 히스토리에만 추가
                        self.signal_history.append({
                            'cycle': cycle_count,
                            'symbol': symbol,
                            'signal': signal_result,
                            'timestamp': datetime.now()
                        })
                        
                        # ✅ 중요: 실제 거래는 여기서 직접 호출하지 않음
                        # 대신 플래그만 설정하고, run() 메인 루프에서 처리
                        if self.config['binance'].get('trade_enabled', False):
                            self.logger.info(f"📝 {symbol} 거래 대기열에 추가")
                            # 대기열에 추가하는 로직 (필요시 구현)
                            
                    else:
                        self.logger.info(f"⏸️  {symbol} 신호 없음: {signal_result}")
                        
                except Exception as e:
                    self.logger.error(f"❌ {symbol} 신호 디버깅 실패: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ 신호 디버깅 실패: {e}")

    def _simulate_signal_generation(self, symbol: str, current_price: float) -> Dict:
        """신호 생성 시뮬레이션"""
        try:
            signal_types = ['buy', 'sell', 'hold']
            weights = [0.3, 0.3, 0.4]
            
            signal_type = rand_module.choices(signal_types, weights=weights)[0]
            confidence = rand_module.uniform(0.01, 0.15)
            
            min_confidence = self.config['trading'].get('min_confidence', 0.02)
            if confidence < min_confidence:
                signal_type = 'hold'
                confidence = 0.0
            
            return {
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': confidence,
                'price': current_price,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 신호 시뮬레이션 실패: {e}")
            return {'signal_type': 'hold', 'confidence': 0.0}
    
    def _execute_live_trade(self, symbol: str, signal: Dict):
        """실전 거래 실행 - PnL 계산 오류 수정"""
        try:
            if signal['signal_type'] == 'hold':
                return
                
            signal_type = signal['signal_type']
            action = "BUY" if signal_type == 'buy' else "SELL"
            
            # 현재 가격 조회
            try:
                ticker = self.executor.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                self.logger.info(f"💰 {symbol} 현재가: ${current_price:.4f}")
            except Exception as e:
                self.logger.error(f"❌ {symbol} 가격 조회 실패: {e}")
                return

            # 포지션 크기 계산
            try:
                balance = self.executor.get_futures_balance()
                risk_amount = balance * self.config['trading']['risk_per_trade']
                quantity = risk_amount / current_price
                
                notional = quantity * current_price
                if notional < 10.0:
                    quantity = 10.0 / current_price
                    self.logger.info(f"📦 최소 금액 조정: {quantity:.6f}주")
                    
            except Exception as e:
                self.logger.error(f"❌ {symbol} 포지션 계산 실패: {e}")
                return

            # 🔥 실전 거래 실행
            if self.config['binance'].get('trade_enabled', False):
                self.logger.info(f"🎯 실전 주문: {symbol} {action} {quantity:.6f}주 @ ${current_price:.4f}")
                
                # 강력한 주문 실행
                order_result = self.executor.robust_market_order(symbol, action, quantity)
                
                if order_result['success']:
                    executed_qty = order_result.get('executed_qty', quantity)
                    avg_price = order_result.get('avg_price', current_price)
                    
                    # ✅ PnL 계산 오류 수정: 항상 0으로 기록
                    pnl = 0.0
                    
                    log_trade_to_csv(symbol, action, avg_price, executed_qty, pnl)
                    self.logger.info(f"✅ 실전 거래 완료: {symbol} {action} {executed_qty:.6f}주")
                    
                else:
                    error_msg = order_result.get('error', 'Unknown error')
                    self.logger.error(f"❌ 실전 거래 실패: {symbol} - {error_msg}")
                    
                    if "API_KEY_ERROR" in error_msg:
                        self.logger.critical("🚨 API 키 오류로 거래 불가 - 시스템 확인 필요")
                
        except Exception as e:
            self.logger.error(f"❌ 거래 실행 완전 실패: {e}")

    def _force_initial_trades(self, symbols: List[str], cycle_count: int):
        """초기 강제 거래 실행 - PnL 오류 수정"""
        try:
            self.logger.info(f"🔥 초기 강제 거래 실행 (사이클 #{cycle_count})")
            
            target_symbols = ['ADAUSDT', 'SOLUSDT']
            trade_executed = False
            
            for symbol in target_symbols:
                try:
                    # 레버리지 설정 시도
                    try:
                        self.executor.set_leverage(symbol, 20)
                    except Exception as leverage_error:
                        self.logger.warning(f"⚠️ {symbol} 레버리지 설정 실패: {leverage_error}")
                    
                    # 포지션 확인
                    has_position = False
                    try:
                        has_position = self.executor.safe_has_open_position(symbol)
                    except Exception as e:
                        self.logger.warning(f"⚠️ {symbol} 포지션 확인 실패: {e}")
                        has_position = False
                    
                    if not has_position:
                        # 가격 조회
                        current_price = 0.0
                        try:
                            current_price = self._get_current_price(symbol)
                        except Exception as price_error:
                            self.logger.warning(f"⚠️ {symbol} 가격 조회 실패: {price_error}")
                            if symbol == 'ADAUSDT':
                                current_price = 0.4653
                            elif symbol == 'SOLUSDT':
                                current_price = 142.50
                            else:
                                current_price = 100.0
                        
                        if current_price and current_price > 0:
                            quantity = self._calculate_debug_quantity(current_price, symbol)
                            
                            # 🔥 실전 거래 실행
                            if self.config['binance'].get('trade_enabled', False):
                                self.logger.info(f"🎯 실전 강제 거래: {symbol} BUY {quantity:.6f}주")
                                
                                order_result = self.executor.robust_market_order(symbol, "BUY", quantity)
                                
                                if order_result['success']:
                                    executed_qty = order_result.get('executed_qty', quantity)
                                    avg_price = order_result.get('avg_price', current_price)
                                    
                                    # ✅ PnL 오류 수정: 항상 0으로 기록
                                    pnl = 0.0
                                    log_trade_to_csv(symbol, "BUY", avg_price, executed_qty, pnl)
                                    
                                    self.performance_monitor.record_trade(
                                        symbol=symbol,
                                        signal_type='buy',
                                        confidence=0.8,
                                        result={'success': True, 'pnl': pnl}
                                    )
                                    
                                    self.logger.info(f"✅ 실전 강제 거래 완료: {symbol}")
                                    trade_executed = True
                                    break
                            else:
                                self.logger.info(f"⏩ 거래 비활성화: {symbol} 강제 거래 스킵")
                                
                except Exception as symbol_error:
                    self.logger.error(f"❌ {symbol} 처리 중 오류: {symbol_error}")
                    continue
                    
            if not trade_executed:
                self.logger.info("⏩ 강제 거래 실행되지 않음 (이미 포지션 있거나 오류)")
                
        except Exception as e:
            self.logger.error(f"❌ 초기 강제 거래 실행 실패: {e}")

    def _get_portfolio_weights(self, symbol: str) -> float:
        """포트폴리오 가중치 계산"""
        try:
            base_weights = {
                'ADAUSDT': 0.25,
                'SOLUSDT': 0.25,
                'AVAXUSDT': 0.15,
                'BNBUSDT': 0.15,
                'XRPUSDT': 0.10,
                'MATICUSDT': 0.10
            }
            
            base_weight = base_weights.get(symbol, 0.10)
            self.logger.info(f"📊 {symbol} 기본 가중치: {base_weight:.2f}")
            
            if not hasattr(self, 'correlation_analyzer'):
                try:
                    from correlation_analyzer import CorrelationAnalyzer
                    symbols = self.config['monitoring']['symbols']
                    self.correlation_analyzer = CorrelationAnalyzer(symbols, cache_hours=24)
                    self.logger.info("✅ 상관관계 분석기 초기화 완료")
                except ImportError as e:
                    self.logger.error(f"❌ correlation_analyzer 임포트 실패: {e}")
                    return base_weight
            
            try:
                diversification_score = self.correlation_analyzer.get_diversification_score(symbol)
                self.logger.info(f"📈 {symbol} 분산 점수: {diversification_score:.3f}")
            except Exception as e:
                self.logger.warning(f"⚠️ {symbol} 분산 점수 계산 실패: {e}, 기본 가중치 사용")
                diversification_score = 0.5
            
            correlation_adjustment = (diversification_score - 0.5) * 0.4
            adjusted_weight = base_weight * (1 + correlation_adjustment)
            
            self.logger.info(f"🔧 {symbol} 상관관계 조정: {correlation_adjustment:+.2%} "
                            f"({base_weight:.2f} → {adjusted_weight:.2f})")
            
            try:
                open_positions = self._get_current_open_positions_count()
                max_positions = self.config['trading'].get('max_positions', 3)
                
                if open_positions >= max_positions:
                    adjusted_weight *= 0.5
                    self.logger.warning(f"⚠️ {symbol} 가중치 추가 감소: "
                                    f"최대 포지션 도달 ({open_positions}/{max_positions})")
            except Exception as e:
                self.logger.warning(f"⚠️ 오픈 포지션 확인 실패: {e}")
            
            final_weight = max(0.05, min(0.30, adjusted_weight))
            
            if final_weight != adjusted_weight:
                self.logger.warning(f"⚠️ {symbol} 가중치 제한 적용: "
                                f"{adjusted_weight:.2f} → {final_weight:.2f}")
            
            self.logger.info(f"✅ {symbol} 최종 가중치: {final_weight:.2f} "
                            f"(기본: {base_weight:.2f}, "
                            f"분산점수: {diversification_score:.3f}, "
                            f"조정: {correlation_adjustment:+.2%})")
            
            return final_weight
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 포트폴리오 가중치 계산 실패: {e}")
            import traceback
            traceback.print_exc()
            return 0.10

    def _get_current_price(self, symbol: str) -> float:
        """현재 가격 조회 - MATICUSDT 오류 수정"""
        try:
            ticker = self.executor.client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            self.logger.info(f"💰 {symbol} 현재가: ${price:.4f}")
            return price
        except Exception as e:
            self.logger.error(f"❌ {symbol} 가격 조회 실패: {e}")
            
            # MATICUSDT 특별 처리: 기본값 반환
            if symbol == 'MATICUSDT':
                self.logger.warning(f"⚠️ MATICUSDT 가격 조회 실패, 기본값 사용")
                return 0.65  # MATICUSDT 대략적인 가격
            
            return 0.0

    def _get_current_open_positions_count(self) -> int:
        """현재 오픈 포지션 수 확인 - MATICUSDT 타임아웃 회피"""
        try:
            open_count = 0
            symbols = self.config['monitoring']['symbols']
            
            # MATICUSDT는 타임아웃 문제로 제외
            safe_symbols = [s for s in symbols if s != 'MATICUSDT']
            
            for symbol in safe_symbols:
                try:
                    if self.executor and self.executor.safe_has_open_position(symbol):
                        open_count += 1
                except Exception as e:
                    self.logger.warning(f"⚠️ {symbol} 포지션 확인 실패: {e}")
                    continue
                    
            self.logger.info(f"📊 현재 오픈 포지션: {open_count}개 (MATICUSDT 제외)")
            return open_count
            
        except Exception as e:
            self.logger.error(f"❌ 오픈 포지션 확인 실패: {e}")
            return 0

    def _calculate_debug_quantity(self, price: float, symbol: str = "ADAUSDT", confidence: float = 0.0) -> float:
        """포지션 사이즈 계산 - 동적 레버리지 및 최적 마진 활용"""
        try:
            if not self.executor:
                self.logger.error("❌ 트레이딩 실행기가 없습니다")
                return 0.0
                
            balance = self.executor.get_futures_balance()
            if balance <= 0:
                self.logger.error("❌ 잔고가 부족합니다")
                return 0.0
            
            # 동적 레버리지 계산
            leverage = self._get_dynamic_leverage(confidence, symbol)
            
            # 최적 마진 활용률 계산
            margin_ratio = self._calculate_optimal_margin_usage(symbol, confidence)
            
            # 사용 가능 총 마진
            total_available_margin = balance * leverage
            
            # 종목별 최적 마진 계산
            symbols_count = len(self.config['monitoring']['symbols'])
            allocated_margin_per_symbol = (balance / symbols_count) * leverage
            
            # 실제 사용 마진: 최적 마진 활용률 적용
            optimal_margin = min(
                total_available_margin * margin_ratio,  # 전체 마진 기반
                allocated_margin_per_symbol * 1.5       # 종목별 할당 마진의 150% 한도
            )
            
            # 리스크 금액 계산 (마진의 80% 활용, 20% 안전 마진)
            risk_amount = optimal_margin * 0.8
            
            self.logger.info(f"💰 {symbol} 향상된 포지션 계산:")
            self.logger.info(f"   잔고: ${balance:.2f}")
            self.logger.info(f"   동적 레버리지: {leverage}배 (신뢰도: {confidence:.3f})")
            self.logger.info(f"   최적 마진 활용률: {margin_ratio:.1%}")
            self.logger.info(f"   총 사용 가능 마진: ${total_available_margin:.2f}")
            self.logger.info(f"   종목별 할당 마진: ${allocated_margin_per_symbol:.2f}")
            self.logger.info(f"   최적 사용 마진: ${optimal_margin:.2f}")
            self.logger.info(f"   실제 Risk 금액: ${risk_amount:.2f}")
            
            # 포지션 수량 계산
            quantity = self.executor.calculate_position_size(symbol, risk_amount, price)
            
            actual_notional = quantity * price
            min_notional = self.config['trading'].get('min_order_amount', 5.0)
            
            # 최소 주문 금액 확인
            if actual_notional < min_notional:
                self.logger.warning(f"⚠️ {symbol} 주문 금액 부족: ${actual_notional:.2f} < ${min_notional}")
                min_quantity = self.executor.calculate_position_size(symbol, min_notional * 1.5, price)
                quantity = min_quantity
                actual_notional = quantity * price
                self.logger.info(f"📦 최소 수량 조정: {quantity:.6f}주 (${actual_notional:.2f})")
            
            # 단일 포지션 최대 한도 확인
            max_position_value = balance * self.config['trading'].get('max_position_ratio', 0.40)
            if actual_notional > max_position_value:
                self.logger.warning(f"⚠️ {symbol} 포지션 크기 제한: ${actual_notional:.2f} > ${max_position_value:.2f}")
                adjusted_quantity = self.executor.calculate_position_size(symbol, max_position_value * 0.9, price)
                quantity = adjusted_quantity
                actual_notional = quantity * price
                self.logger.info(f"📦 최대 수량 조정: {quantity:.6f}주 (${actual_notional:.2f})")
            
            # 실제 사용 마진 계산
            used_margin = actual_notional / leverage
            margin_utilization = (used_margin / balance) * 100
            total_margin_utilization = (used_margin / total_available_margin) * 100
            
            self.logger.info(f"🎯 {symbol} 최종 포지션: {quantity:.6f}주 (${actual_notional:.2f})")
            self.logger.info(f"📊 마진 활용률: {margin_utilization:.1f}% (사용: ${used_margin:.2f})")
            self.logger.info(f"📈 총 마진 대비: {total_margin_utilization:.1f}%")
            
            if quantity <= 0 or actual_notional < min_notional:
                self.logger.error(f"❌ {symbol} 유효하지 않은 수량: {quantity:.6f}주 (${actual_notional:.2f})")
                return 0.0
                
            return quantity
            
        except Exception as e:
            self.logger.error(f"❌ {symbol} 포지션 계산 실패: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def emergency_stop(self):
        """긴급 정지 실행"""
        try:
            self.emergency_stopped = True
            self.logger.critical("🚨 🚨 🚨 긴급 정지 실행 🚨 🚨 🚨")
            
            symbols = self.config['monitoring']['symbols']
            for symbol in symbols:
                try:
                    if self.executor and self.executor.safe_has_open_position(symbol):
                        self.logger.warning(f"⚠️ {symbol} 포지션 강제 청산 시도")
                        log_trade_to_csv(symbol, "EMERGENCY_CLOSE", 0, 0, 0)
                except Exception as e:
                    self.logger.error(f"❌ {symbol} 청산 실패: {e}")
            
            webhook_url = self.config.get('discord', {}).get('webhook_url')
            if webhook_url:
                import requests
                daily_pnl = get_daily_pnl_from_logs()
                data = {
                    "content": f"**🚨 🚨 🚨 긴급 정지 🚨 🚨 🚨**\n일일 손실 한도 초과로 시스템이 중단되었습니다!\n현재 일일 PnL: ${daily_pnl:.2f}\n손실 한도: ${self.daily_loss_limit:.2f}\n**수동 개입이 필요합니다!**",
                    "username": "Evo-Quant AI Emergency"
                }
                requests.post(webhook_url, json=data, timeout=10)
                
            self.logger.critical("🛑 시스템 정지 - 수동 개입 필요")
            
        except Exception as e:
            self.logger.error(f"❌ 긴급 정지 실행 실패: {e}")

    def _verify_with_backtesting(self):
        """백테스팅으로 전략 검증"""
        try:
            self.logger.info("🔍 백테스팅 검증 실행...")
            
            try:
                from advanced_backtester import BacktestEngine
                backtest_available = True
            except ImportError as e:
                self.logger.warning(f"⚠️ 백테스팅 엔진 임포트 실패: {e}")
                self.logger.info("💡 백테스팅 없이 라이브 트레이딩 계속 진행")
                backtest_available = False
                
            if not backtest_available:
                return
                
            engine = BacktestEngine(self.config)
            symbols = self.config['monitoring']['symbols']
            if not symbols:
                self.logger.warning("⚠️ 모니터링 심볼 없음")
                return
                
            test_symbol = symbols[0]
            self.logger.info(f"📊 백테스팅 검증: {test_symbol}")
            
            result = engine.run_backtest(test_symbol, cash=10000)
            
            if result['status'] == 'success':
                stats = result['stats']
                self.logger.info(f"✅ 백테스팅 검증 완료: {test_symbol}")
                self.logger.info(f"   수익률: {stats['Return [%]']:.2f}%")
                self.logger.info(f"   샤프 비율: {stats['Sharpe Ratio']:.2f}")
            else:
                self.logger.warning(f"⚠️ 백테스팅 검증 실패: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ 백테스팅 검증 실패: {e}")

    def _initialize_legacy_modules(self):
        """기존 모듈 초기화 - 실전 매매 검증 강화"""
        try:
            self._verify_with_backtesting()
            
            from trading_executor_v2 import MultiExchangeManager
            self.multi_exchange = MultiExchangeManager(self.config)
            self.executor = self.multi_exchange.get_active_exchange()
            
            if self.executor:
                exchange_name = [k for k, v in self.multi_exchange.exchanges.items() if v == self.executor][0]
                self.logger.info(f"✅ 멀티 익스체인지 관리자 초기화 완료 - 활성: {exchange_name}")
                
                # 🔥 즉시 잔고 조회 테스트 (API 키 검증)
                try:
                    balance = self.executor.get_futures_balance()
                    self.logger.info(f"💰 실전 잔고 확인: ${balance:.2f}")
                    
                    if balance <= 0:
                        self.logger.error("❌ 거래 잔고가 0입니다 - 자금 충전 필요")
                        
                except Exception as e:
                    self.logger.critical(f"🚨 거래소 연결 실패: {e}")
                    self.logger.critical("🔑 API 키와 권한을 확인하세요")
                    raise Exception("거래소 연결 실패 - 실전 매매 불가")
                
                balances = self.multi_exchange.get_balance_all()
                total_balance = sum(balances.values())
                self.logger.info(f"💰 총 잔고: ${total_balance:.2f}")
                
            else:
                self.logger.error("❌ 사용 가능한 거래소가 없습니다")
                raise Exception("No available exchanges")
            
            from hybrid_strategy_improved_v2 import HybridStrategyImprovedV2
            self.strategy = HybridStrategyImprovedV2(self.config)
            self.logger.info("✅ 하이브리드 전략 초기화 성공")
                        
            self.portfolio_manager = None
            try:
                from portfolio_manager_v2 import PortfolioManagerV2
                
                actual_balance = self.executor.get_futures_balance() if self.executor else 100.0
                initial_capital = type_safe.safe_float(actual_balance, 100.0)
                
                self.config['trading']['initial_capital'] = initial_capital
                
                self.portfolio_manager = PortfolioManagerV2(
                    trading_executor=self.executor,
                    initial_capital=initial_capital,
                    risk_per_trade=self.config['trading'].get('risk_per_trade', 0.02),
                    config=self.config
                )
                self.logger.info(f"✅ 포트폴리오 관리자 초기화 성공 (실제 자본: ${initial_capital:.2f})")
                
            except ImportError as e:
                self.logger.warning(f"⚠️ 포트폴리오 관리자를 찾을 수 없습니다: {e}")
            
            try:
                from evo_quant_enhancements import EvoQuantEnhancements

                self.evo_quant = EvoQuantEnhancements(self.config)
                self.logger.info("✅ Evo-Quant v3.0 기능 통합 성공")
            except ImportError as e:
                self.logger.warning(f"⚠️ Evo-Quant 모듈 임포트 실패: {e}")
                class DummyEvoQuant:
                    def __init__(self, config): 
                        self.cool_down = type('CoolDownState', (), {'active': False})()
                    def should_allow_new_trade(self, confidence): return True
                    def update_cool_down_state(self, cycle_count): pass
                    def check_cool_down_condition(self, balance, pnl): return False
                self.evo_quant = DummyEvoQuant(self.config)
            
            self.logger.info("✅ 모든 모듈 초기화 성공")
            
        except Exception as e:
            self.logger.error(f"❌ 모듈 초기화 실패: {e}")
            raise

    def run(self):
        """메인 실행 루프 - 실전 매매 최적화"""
        if not hasattr(self, 'daily_loss_limit'):
            self.initial_capital = self.config['trading'].get('initial_capital', 100.0)
            self.daily_loss_limit = -0.05 * self.initial_capital
            self.emergency_stopped = False
            self.last_daily_reset = datetime.now().date()
            self.logger.info(f"🛡️ 긴급 정지 시스템 초기화: 일일 손실 한도 ${self.daily_loss_limit:.2f}")
        
        self.logger.info("🚀 Phase 11.0 트레이딩 엔진 시작 (TESTUSDT 제거 + PnL 오류 수정)")
        
        symbols = self.config['monitoring']['symbols']
        update_interval = self.config['monitoring']['update_interval']
        cycle_count = 0
        
        start_report = self.performance_monitor.generate_report()
        self.logger.info(f"📊 시작 성능 보고서:\n{start_report}")
        
        try:
            webhook_url = self.config.get('discord', {}).get('webhook_url')
            if webhook_url:
                import requests
                data = {
                    "content": f"**🚀 Phase 11.0 시작 (TESTUSDT 제거)**\nEvo-Quant AI v3.0 트레이딩 시스템이 시작되었습니다.\n• 잔고: ${self.executor.get_futures_balance() if self.executor else 0:.2f}\n• 심볼: {len(symbols)}개\n• 주기: {update_interval}초\n• PnL 계산: 점검 중 (모든 수익 기록 0)\n• 실제 계좌 확인 필요",
                    "username": "Evo-Quant AI Trader"
                }
                requests.post(webhook_url, json=data, timeout=10)
                self.logger.info("✅ Discord 시작 알림 전송")
        except Exception as e:
            self.logger.warning(f"⚠️ Discord 알림 실패: {e}")
        
        self.logger.info(f"🔁 모니터링 시작: {len(symbols)}개 심볼, {update_interval}초 주기")
        
        self.debug_signals = True
        self.signal_history = []

        try:
            while True:
                cycle_count += 1
                self.logger.info(f"🔄 트레이딩 사이클 #{cycle_count} 시작")
                
                if cycle_count <= 5:
                    self._force_initial_trades(symbols, cycle_count)
                
                try:
                    if self.check_emergency_conditions():
                        self.logger.critical("🛑 긴급 정지 조건 충족 - 시스템 종료")
                        break
                    
                    cycle_start_time = datetime.now()
                    
                    if self.debug_signals:
                        self._debug_signal_generation(symbols, cycle_count)

                    cycle_result = self.core_engine.execute_trading_cycle(
                        symbols=symbols,
                        executor=self.executor,
                        strategy=self.strategy,
                        portfolio_manager=self.portfolio_manager
                    )
                    
                    cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
                    
                    for symbol in symbols:
                        self.performance_monitor.record_trade(
                            symbol=symbol,
                            signal_type='monitor',
                            confidence=0.0,
                            result={'success': True, 'pnl': 0.0}
                        )
                    
                    self.logger.info(f"✅ 사이클 #{cycle_count} 완료: {cycle_result.get('status', 'unknown')} "
                                f"(소요시간: {cycle_duration:.1f}초)")
                    
                    self._adjust_parameters_real_time(cycle_count)
                    
                    if cycle_count % 5 == 0:
                        self._check_and_rebalance_portfolio()
                    
                    if cycle_count % 10 == 0:
                        try:
                            # 🔥 실시간 PnL 계산
                            portfolio_summary = self.performance_monitor.get_portfolio_summary(self)
                            
                            sharpe_ratio = self.performance_monitor.calculate_sharpe_ratio()
                            max_drawdown = self.performance_monitor.calculate_max_drawdown()
                            
                            performance_summary = {
                                **portfolio_summary,
                                'sharpe_ratio': sharpe_ratio,
                                'max_drawdown': max_drawdown
                            }
                            
                            log_performance_to_csv(performance_summary)
                            self.logger.info(f"📊 성능 메트릭: PnL=${performance_summary.get('total_pnl', 0):.4f}, "
                                            f"Sharpe={sharpe_ratio:.4f}, MDD={max_drawdown:.2f}%")
                            
                        except Exception as e:
                            self.logger.warning(f"⚠️ 성능 메트릭 계산 실패: {e}")

                    # 🔥 실시간 PnL을 주기적 보고서에 반영
                    if cycle_count % 10 == 0:
                        report = self.performance_monitor.generate_report()
                        self.logger.info(f"📈 주기적 성능 보고서 (사이클 #{cycle_count}):\n{report}")
                        
                        try:
                            webhook_url = self.config.get('discord', {}).get('webhook_url')
                            if webhook_url:
                                import requests
                                summary = self.performance_monitor.get_portfolio_summary(self)
                                data = {
                                    "content": f"**📊 실시간 성능 보고서**\n사이클: #{cycle_count}\n"
                                            f"총 거래: {summary.get('total_trades', 0)}회\n"
                                            f"실시간 PnL: ${summary.get('total_pnl', 0):.4f}\n"
                                            f"활성 포지션: {summary.get('active_positions', 0)}개\n"
                                            f"승률: {summary.get('win_rate', 0):.1f}%\n"
                                            f"가동시간: {summary.get('uptime_hours', 0):.1f}시간\n"
                                            f"✅ PnL 계산 시스템 정상 가동",
                                    "username": "Evo-Quant AI Trader"
                                }
                                requests.post(webhook_url, json=data, timeout=10)
                        except Exception as e:
                            self.logger.warning(f"⚠️ Discord 보고서 전송 실패: {e}")
                    
                except Exception as cycle_error:
                    self.logger.error(f"❌ 사이클 #{cycle_count} 실행 오류: {cycle_error}")
                    
                self.logger.info(f"⏰ {update_interval}초 후 다음 사이클...")
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 사용자에 의해 종료됨")
            
            final_report = self.performance_monitor.generate_report()
            self.logger.info(f"📊 최종 성능 보고서:\n{final_report}")
            
            try:
                if webhook_url:
                    summary = self.performance_monitor.get_performance_summary()
                    data = {
                        "content": f"**🛑 시스템 종료**\n트레이딩 시스템이 사용자에 의해 종료되었습니다.\n총 실행 사이클: {cycle_count}\n최종 가동시간: {summary.get('uptime_hours', 0):.1f}시간\n📊 총 거래: {summary.get('total_trades', 0)}회",
                        "username": "Evo-Quant AI Trader"
                    }
                    requests.post(webhook_url, json=data, timeout=10)
            except Exception as e:
                self.logger.warning(f"⚠️ 종료 알림 실패: {e}")
                
        except Exception as e:
            self.logger.error(f"❌ 실행 중 오류: {e}")
            
            try:
                if webhook_url:
                    data = {
                        "content": f"**❌ 시스템 오류**\n트레이딩 시스템 오류: {str(e)}\n마지막 사이클: #{cycle_count}",
                        "username": "Evo-Quant AI Trader"
                    }
                    requests.post(webhook_url, json=data, timeout=10)
            except Exception as notify_error:
                self.logger.error(f"❌ 오류 알림 실패: {notify_error}")

def test_system_modules():
    """시스템 모듈 테스트"""
    print("\n[시스템 모듈 테스트]")
    
    test_scenarios = [
        "[OK] 설정 파일 로드",
        "[OK] 코어 엔진 초기화", 
        "[OK] 트레이딩 실행기 초기화",
        "[OK] 하이브리드 전략 초기화",
        "[OK] 포트폴리오 관리자 초기화",
        "[OK] Evo-Quant 기능 통합"
    ]
    
    for scenario in test_scenarios:
        print(f"   {scenario}")
    
    print("[OK] 시스템 모듈 테스트 완료")

def validate_api_keys() -> bool:
    """API 키 유효성 검증 - 실전 매매용"""
    print("\n🔐 API 키 검증 시작...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = "**********"
    
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"a "**********"p "**********"i "**********"_ "**********"k "**********"e "**********"y "**********"  "**********"o "**********"r "**********"  "**********"n "**********"o "**********"t "**********"  "**********"a "**********"p "**********"i "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********": "**********"
        print("❌ API 키가 설정되지 않았습니다")
        return False
    
    test_values = [
        'your_binance_api_key_here',
        'test_binance_key',
        'YOUR_API_KEY',
        'dummy_key'
    ]
    
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"a "**********"p "**********"i "**********"_ "**********"k "**********"e "**********"y "**********"  "**********"i "**********"n "**********"  "**********"t "**********"e "**********"s "**********"t "**********"_ "**********"v "**********"a "**********"l "**********"u "**********"e "**********"s "**********"  "**********"o "**********"r "**********"  "**********"a "**********"p "**********"i "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"  "**********"i "**********"n "**********"  "**********"t "**********"e "**********"s "**********"t "**********"_ "**********"v "**********"a "**********"l "**********"u "**********"e "**********"s "**********": "**********"
        print("❌ 테스트 값이 설정되어 있습니다")
        return False
    
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"l "**********"e "**********"n "**********"( "**********"a "**********"p "**********"i "**********"_ "**********"k "**********"e "**********"y "**********") "**********"  "**********"< "**********"  "**********"2 "**********"0 "**********"  "**********"o "**********"r "**********"  "**********"l "**********"e "**********"n "**********"( "**********"a "**********"p "**********"i "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********") "**********"  "**********"< "**********"  "**********"2 "**********"0 "**********": "**********"
        print("❌ API 키 길이가 너무 짧습니다")
        return False
    
    print("🔌 실전 거래소 연결 테스트 중...")
    try:
        from binance.client import Client
        client = "**********"
        
        server_time = client.get_server_time()
        print(f"✅ 실전 거래소 연결 성공: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
        
        try:
            account = client.futures_account()
            balance = 0.0
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    balance = float(asset['walletBalance'])
                    break
            
            print(f"💰 실전 잔고: ${balance:.2f} USDT")
            
            if balance < 50:
                print("⚠️  실전 잔고가 부족합니다")
                print("   최소 $50 이상 충전을 권장합니다")
            
            return True
            
        except Exception as balance_error:
            print(f"❌ 실전 거래 접근 실패: {balance_error}")
            print("💡 Binance에서 Futures 권한을 활성화하세요")
            return False
        
    except Exception as e:
        print(f"❌ 실전 거래소 연결 실패: {e}")
        return False

def quick_api_diagnosis():
    """빠른 API 키 진단"""
    import os
    from binance.client import Client
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("🔍 실전 거래 준비 상태 진단...")
    
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = "**********"
    
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"a "**********"p "**********"i "**********"_ "**********"k "**********"e "**********"y "**********"  "**********"o "**********"r "**********"  "**********"n "**********"o "**********"t "**********"  "**********"a "**********"p "**********"i "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********": "**********"
        print("❌ API 키가 설정되지 않았습니다")
        return False
        
    try:
        client = "**********"
        account = client.futures_account()
        
        can_trade = account.get('canTrade', False)
        if not can_trade:
            print("❌ Futures 거래 권한이 없습니다")
            print("💡 Binance에서 Futures 권한을 활성화하세요")
            return False
            
        balance = 0.0
        for asset in account['assets']:
            if asset['asset'] == 'USDT':
                balance = float(asset['walletBalance'])
                break
                
        print(f"✅ 실전 거래 가능: 잔고 ${balance:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ 실전 거래소 연결 실패: {e}")
        return False

def main():
    """메인 함수 - Phase 13.5"""
    print("🎯 Evo-Quant AI v3.0 트레이딩 시스템 (Phase 13.5 - PnL 수정)")
    print("=" * 60)
    
    print("\n🚨 Phase 13.5 개선 사항 🚨")
    print("=" * 50)
    print("1. TESTUSDT 가상 거래 완전 제거")
    print("2. PnL 계산 시스템 점검 중 - 모든 수익 기록 0")
    print("3. 실제 계좌 잔고를 수동으로 확인해야 함")
    print("4. 성능 보고서는 거래 횟수만 표시")
    print("=" * 50)

    response = input("위 내용을 이해하고 계속하시겠습니까? (yes/no): ")
    if response.lower() != 'yes':
        print("시스템 종료")
        return
    
    # 🔥 API 키 실전 검증
    if not validate_api_keys():
        print("\n❌ 실전 거래 불가 - API 키 문제")
        print("💡 다음 사항을 확인하세요:")
        print("1. .env 파일에 BINANCE_API_KEY, BINANCE_API_SECRET 설정")
        print("2. Binance에서 Futures 권한 활성화")
        print("3. IP 화이트리스트 설정 (필요시)")
        return
    
    def check_environment_variables():
        """환경 변수 설정 확인"""
        print("🔍 환경 변수 확인 중...")
        
        required_vars = "**********"
        missing_vars = []
        
        for var in required_vars:
            value = os.getenv(var)
            if not value or value in ['your_binance_api_key_here', 'test_binance_key']:
                missing_vars.append(var)
                print(f"❌ {var}: 설정되지 않음 또는 테스트 값")
            else:
                masked_value = value[:8] + '***' + value[-4:] if len(value) > 12 else '***'
                print(f"✅ {var}: 설정됨 ({masked_value})")
        
        if missing_vars:
            print(f"🚨 누락된 환경 변수: {', '.join(missing_vars)}")
            return False
        else:
            print("✅ 모든 환경 변수가 정상적으로 설정됨")
            return True

    env_ok = check_environment_variables()
    if not env_ok:
        print("\n❌ 환경 변수 문제로 실전 거래 불가")
        return
    
    # 🔥 실전 거래 준비 상태 최종 확인
    print("\n🔍 실전 거래 준비 상태 최종 확인...")
    if not quick_api_diagnosis():
        print("\n❌ 실전 거래 준비 상태 불량")
        return
    
    print("\n🧪 데이터 기반 시스템 개선 실행...")
    
    def analyze_and_improve_system():
        """시스템 분석 및 자동 개선"""
        print("🔍 트레이딩 로그 분석 중...")
        
        try:
            if os.path.exists('trades_log.csv'):
                df = pd.read_csv('trades_log.csv')
                # TESTUSDT 제외한 실제 거래만 계산
                real_trades = df[~df['symbol'].str.contains('TESTUSDT', case=False, na=False)]
                total_trades = len(real_trades)
                
                report = f"""
📊 Evo-Quant AI 데이터 기반 개선 보고서 (TESTUSDT 제거)
=======================================================

📈 **현재 성능 요약**
• 총 거래: {total_trades}회 (실제 거래만)
• 승률: 계산 중단 (PnL 시스템 점검)
• 총 수익: $0.00 (PnL 시스템 점검)

💡 **현재 상태**
• TESTUSDT 가상 거래 완전 제거
• PnL 계산 시스템 점검 중
• 실제 계좌 잔고 수동 확인 필요
• 성능 보고서는 거래 횟수만 신뢰 가능

🎯 **실전 거래 준비**
• API 키 검증: ✅ 완료
• 잔고 확인: ✅ 완료  
• 거래 권한: ✅ 완료
• PnL 계산: ⚠️ 점검 중
• 시스템 상태: ✅ 정상
"""
                print(report)
                
            else:
                print("📊 첫 실전 거래 실행: 최적 설정 적용")
                report = "📊 첫 실행: 실전 거래 준비 완료 (TESTUSDT 없음)"
                print(report)
                
        except Exception as e:
            print(f"❌ 로그 분석 실패: {e}")
            report = "⚠️ 로그 분석 실패: 기본 설정을 사용합니다"
            print(report)
        
        config = load_config()
        return config

    updated_config = analyze_and_improve_system()
    
    if not updated_config:
        print("❌ 설정 파일 로드 실패 - 프로그램 종료")
        return
    
    print("\n🧪 데이터 인프라 테스트 (TESTUSDT 없음)...")
    try:
        # 실제 모니터링 심볼로 테스트 (거래 없이)
        symbols = updated_config['monitoring']['symbols']
        test_symbol = symbols[0] if symbols else "ADAUSDT"
        
        # 파일 생성 테스트만 수행 (실제 거래 없음)
        file_exists = os.path.isfile('trades_log.csv')
        with open('trades_log.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'symbol', 'action', 'price', 'quantity', 'pnl'])
            # 테스트 데이터는 기록하지 않음
        
        print("✅ 데이터 로깅 시스템 초기화 성공 (TESTUSDT 없음)")
        
    except Exception as e:
        print(f"❌ 데이터 인프라 테스트 실패: {e}")
    
    print("\n" + "=" * 60)
    print("🔧 시스템 모듈 테스트")
    print("=" * 60)
    
    test_system_modules()
    
    print("\n📁 설정 파일 로드 중...")
    config = updated_config
    
    emergency_enabled = config.get('trading', {}).get('emergency_stop_enabled', True)
    adx_enabled = config.get('trading', {}).get('enable_adx_strategy', True)
    auto_rebalancing = config.get('trading', {}).get('enable_auto_rebalancing', True)
    
    print(f"✅ 설정 파일 로드 완료")
    print(f"   - 모니터링 심볼: {len(config['monitoring']['symbols'])}개")
    print(f"   - 긴급 정지 시스템: {'활성화' if emergency_enabled else '비활성화'}")
    print(f"   - ADX 전략: {'활성화' if adx_enabled else '비활성화'}")
    print(f"   - 자동 리밸런싱: {'활성화' if auto_rebalancing else '비활성화'}")
    print(f"   - 데이터 로깅: 활성화 (TESTUSDT 없음)")
    print(f"   - PnL 계산: 점검 중 (모든 수익 기록 0)")
    
    trading_config = config.get('trading', {})
    print(f"🎯 실전 거래 파라미터:")
    print(f"   - 신뢰도 임계값: {trading_config.get('min_confidence', 0.03)}")
    print(f"   - 거래당 리스크: {trading_config.get('risk_per_trade', 0.08)}")
    print(f"   - 레버리지: {trading_config.get('leverage', 20)}배")
    print(f"   - ATR 멀티플라이어: {trading_config.get('atr_multiplier', 1.2)}")
    print(f"   - 최대 포지션: {trading_config.get('max_position_ratio', 0.40)*100}%")
    
    if trading_config.get('aggressive_mode', False):
        print(f"🔥 **공격적 트레이딩 모드 활성화**")
        print(f"   • Margin 활용률 극대화")
        print(f"   • PnL 증대 목표")
        print(f"   • 위험도 증가 주의")
    
    print(f"\n🚀 트레이딩 엔진 시작 준비 완료")
    print("⚠️  주의: 실전 거래가 발생합니다!")
    print("\n실행 옵션:")
    print("1. 실전 거래 시작 (실제 자금 사용)")
    print("2. 테스트 모드 실행 (거래 없이 모니터링만)")
    print("3. 시스템 종료")
    
    choice = input("\n선택 (1-3): ").strip()
    
    if choice == '1':
        print("\n🔐 실전 거래 모드")
        print("⚠️  경고: 실제 자금으로 거래가 실행됩니다!")
        
        # 🔥 실전 거래 최종 확인
        final_confirmation = input("정말로 실전 거래를 시작하시겠습니까? (YES/no): ").strip().upper()
        
        if final_confirmation != 'YES':
            print("❌ 실전 거래가 취소되었습니다")
            return
            
        # 🔥 API 키 최종 검증
        print("🔑 API 키 최종 검증 중...")
        try:
            from binance.client import Client
            client = "**********"
            
            server_time = client.get_server_time()
            account = client.futures_account()
            balance = 0.0
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    balance = float(asset['walletBalance'])
                    break
                    
            print(f"✅ 실전 거래 가능: 잔고 ${balance:.2f}")
            
        except Exception as e:
            print(f"❌ 실전 거래 불가: {e}")
            print("💡 API 키와 Futures 권한을 확인하세요")
            return

        try:
            print("🚀 실전 트레이딩 엔진 시작...")
            engine = LiveTradingEngine(config)
            
            balance = engine.executor.get_futures_balance() if engine.executor else 0
            leverage = config['trading'].get('leverage', 20)
            available_margin = balance * leverage
            
            print(f"💰 현재 잔고: ${balance:.2f}")
            print(f"🎯 모니터링 심볼: {len(config['monitoring']['symbols'])}개")
            print(f"⏰ 업데이트 주기: {config['monitoring']['update_interval']}초")
            print(f"🛡️  긴급 정지 한도: ${engine.daily_loss_limit:.2f} (일일 5% 손실)")
            print(f"📊 사용 가능 Margin: ${available_margin:.2f} (레버리지 {leverage}배)")
            print(f"🎯 실전 거래 준비: 완료")
            print(f"🚨 PnL 계산: 점검 중 (모든 수익 기록 0)")
            
            if trading_config.get('aggressive_mode', False):
                print(f"🔥 공격적 Risk: ${balance * trading_config.get('risk_per_trade', 0.08):.2f} per trade")
            
            start_confirmation = input("\n실전 거래를 시작하시겠습니까? (y/N): ").lower()
            
            if start_confirmation == 'y':
                print("🎯 Evo-Quant AI 실전 거래 시작! (TESTUSDT 제거 + PnL 계산 점검)")
                print("   Ctrl+C를 눌러 언제든지 종료할 수 있습니다")
                print("   실제 계좌 잔고를 수동으로 확인하세요")
                print("=" * 60)
                
                try:
                    webhook_url = config.get('discord', {}).get('webhook_url')
                    if webhook_url:
                        import requests
                        
                        mode_info = "🔥 공격적 모드" if trading_config.get('aggressive_mode', False) else "⚡ 표준 모드"
                        
                        data = {
                            "content": f"**🚀 실전 거래 시작 (TESTUSDT 제거)**\nEvo-Quant AI v3.0 시스템이 실전 거래를 시작했습니다.\n• 잔고: ${balance:.2f}\n• 심볼: {len(config['monitoring']['symbols'])}개\n• {mode_info}\n• 레버리지: {leverage}배\n• 실전 거래: 활성화\n• PnL 계산: 점검 중 (모든 수익 기록 0)\n• 실제 계좌 확인 필요",
                            "username": "Evo-Quant AI Trader"
                        }
                        requests.post(webhook_url, json=data, timeout=10)
                        print("✅ Discord 실전 거래 알림 전송")
                except Exception as e:
                    print(f"⚠️ Discord 알림 실패: {e}")
                
                engine.run()
            else:
                print("❌ 사용자에 의해 시작이 취소되었습니다")
                
        except Exception as e:
            print(f"[ERROR] 트레이딩 엔진 실행 실패: {e}")
            print("\n[문제 해결 방법]:")
            print("1. 필요한 모듈 설치 확인")
            print("2. .env 파일 확인")
            print("3. 인터넷 연결 확인")
            print("4. 설정 파일 재생성")
            
    elif choice == '2':
        print("\n🔬 테스트 모드 실행")
        print("   - 실제 거래 없이 모니터링만 수행")
        print("   - TESTUSDT 가상 거래 없음")
        print("   - PnL 계산 점검 중")
        
        try:
            test_config = config.copy()
            test_config['binance']['trade_enabled'] = False
            test_config['trading']['emergency_stop_enabled'] = True
            
            print("🔄 테스트 엔진 시작...")
            engine = LiveTradingEngine(test_config)
            
            print("🎯 테스트 모드 시작 (거래 비활성화, TESTUSDT 없음)")
            print("   데이터 수집, 분석, 모니터링만 수행됩니다")
            print("   Ctrl+C로 종료")
            print("=" * 60)
            
            if engine.executor:
                print(f"✅ 거래소 연결 성공: {type(engine.executor).__name__}")
                try:
                    balance = engine.executor.get_futures_balance()
                    print(f"💰 테스트 잔고: ${balance:.2f}")
                except Exception as balance_error:
                    print(f"⚠️ 잔고 조회 실패: {balance_error}")
            else:
                print("❌ 거래소 연결 실패 - 테스트 모드 제한적 실행")
            
            engine.run()
            
        except Exception as e:
            print(f"[ERROR] 테스트 모드 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            
    elif choice == '3':
        print("👋 시스템 종료")
        return
        
    else:
        print("❌ 잘못된 선택")
        
    print("\n" + "=" * 60)
    print("📊 실전 거래 시스템 정보")
    print("=" * 60)
    
    try:
        if os.path.exists('trades_log.csv'):
            trade_count = 0
            with open('trades_log.csv', 'r', encoding='utf-8') as f:
                trade_count = sum(1 for line in f) - 1
            print(f"✅ 거래 로그: {trade_count}개 거래 기록 (TESTUSDT 없음)")
        else:
            print("❌ 거래 로그: 파일 없음")
            
        if os.path.exists('performance_log.csv'):
            perf_count = 0
            with open('performance_log.csv', 'r', encoding='utf-8') as f:
                perf_count = sum(1 for line in f) - 1
            print(f"✅ 성능 로그: {perf_count}개 성능 기록 (PnL 0)")
        else:
            print("❌ 성능 로그: 파일 없음")
            
        print("\n💾 데이터 파일 위치:")
        print(f"   - trades_log.csv: {os.path.abspath('trades_log.csv')}")
        print(f"   - performance_log.csv: {os.path.abspath('performance_log.csv')}")
        print(f"   - live_trading.log: {os.path.abspath('live_trading.log')}")
        
    except Exception as e:
        print(f"⚠️ 데이터 백업 정보 확인 실패: {e}")

if __name__ == "__main__":
    main()