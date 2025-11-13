#date: 2025-11-13T17:03:12Z
#url: https://api.github.com/gists/d68fdd03a1f1096971a5f4ed71530eb2
#owner: https://api.github.com/users/wangwei334455

"""
MT5数据服务 - Windows前置机使用
轻量级TCP Socket服务，最快速度传输MT5数据，不存储任何数据
"""
import sys
import socket
import threading
import time
import json
import struct
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

import MetaTrader5 as mt5
from loguru import logger


class MT5DataService:
    """MT5数据服务 - TCP Socket被动监听，最快速度实时传输"""
    
    def __init__(self, host="0.0.0.0", port=8888):
        self.host = host
        self.port = port
        self.running = False
        self.mt5_connected = False
        self.client_socket = None
        self.client_connected = False
        self.lock = threading.Lock()
        
        logger.info(f"MT5数据服务初始化 - TCP Socket: {host}:{port}")
    
    def connect_mt5(self):
        """连接MT5"""
        try:
            if not mt5.initialize():
                logger.error("MT5初始化失败")
                return False
            
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("无法获取MT5账户信息")
                return False
            
            logger.info(f"MT5连接成功: {account_info.server}")
            self.mt5_connected = True
            return True
        except Exception as e:
            logger.error(f"MT5连接失败: {e}")
            return False
    
    def send_tick_fast(self, tick_data):
        """快速发送TICK数据（二进制格式，最快速度）"""
        if not self.client_connected or not self.client_socket:
            return False
        
        try:
            # 使用二进制格式：时间戳(8字节) + bid(8字节) + ask(8字节) + last(8字节) + volume(8字节)
            # 总共40字节，比JSON快得多
            data = struct.pack('!d d d d d',
                tick_data.get('time_msc', 0) / 1000.0,  # 时间戳（秒）
                tick_data.get('bid', 0.0),
                tick_data.get('ask', 0.0),
                tick_data.get('last', 0.0),
                tick_data.get('volume', 0.0)
            )
            
            with self.lock:
                self.client_socket.sendall(data)
            return True
        except Exception as e:
            logger.error(f"发送数据失败: {e}")
            self.client_connected = False
            return False
    
    def stream_ticks(self, symbol="BTCUSDm"):
        """实时流式传输TICK数据（最快速度）"""
        if not self.mt5_connected:
            if not self.connect_mt5():
                return
        
        logger.info(f"开始高速流式传输 {symbol} TICK数据")
        
        last_time = 0
        count = 0
        
        try:
            while self.client_connected and self.running:
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    # 只发送新数据（避免重复）
                    if tick.time_msc != last_time:
                        tick_data = {
                            'time_msc': tick.time_msc,
                            'bid': tick.bid,
                            'ask': tick.ask,
                            'last': tick.last,
                            'volume': tick.volume
                        }
                        self.send_tick_fast(tick_data)
                        last_time = tick.time_msc
                        count += 1
                
                # 最小延迟：10ms（100Hz）
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"流式传输失败: {e}")
        finally:
            logger.info(f"流式传输停止，共传输 {count} 条数据")
    
    def handle_client(self, client_socket, addr):
        """处理客户端连接"""
        logger.info(f"✅ 客户端连接: {addr}")
        
        with self.lock:
            self.client_socket = client_socket
            self.client_connected = True
        
        # 设置TCP_NODELAY，禁用Nagle算法，降低延迟
        client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        client_socket.settimeout(30)
        
        try:
            # 启动数据流
            stream_thread = threading.Thread(target=self.stream_ticks, args=("BTCUSDm",))
            stream_thread.daemon = True
            stream_thread.start()
            
            # 保持连接，接收心跳
            while self.client_connected:
                try:
                    data = client_socket.recv(1)
                    if not data:
                        logger.info("客户端主动断开")
                        break
                    # 心跳或命令处理（最小化处理）
                except socket.timeout:
                    # 发送心跳保持连接
                    try:
                        client_socket.send(b'\x00')
                    except:
                        logger.warning("发送心跳失败")
                        break
                except Exception as e:
                    logger.error(f"处理客户端数据失败: {e}")
                    break
        except Exception as e:
            logger.error(f"客户端连接处理失败: {e}")
        finally:
            with self.lock:
                self.client_connected = False
                self.client_socket = None
            try:
                client_socket.close()
            except:
                pass
            logger.info(f"客户端断开: {addr}")
    
    def listen_for_clients(self):
        """监听客户端连接"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.settimeout(1)
        
        try:
            sock.bind((self.host, self.port))
            sock.listen(5)
            logger.info(f"✅ 监听端口 {self.port}，等待客户端连接...")
            logger.info(f"   服务地址: {self.host}:{self.port}")
            
            self.running = True
            
            while self.running:
                try:
                    conn, addr = sock.accept()
                    # 设置TCP_NODELAY，最快速度
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    
                    # 处理客户端连接
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(conn, addr)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.timeout:
                    continue
        except Exception as e:
            if self.running:
                logger.error(f"接受连接失败: {e}")
        except OSError as e:
            if e.errno == 98 or e.errno == 10048:  # Address already in use
                logger.error(f"端口 {self.port} 已被占用，请检查是否有其他服务在使用")
            else:
                logger.error(f"监听失败: {e}")
        except Exception as e:
            logger.error(f"监听异常: {e}")
        finally:
            if sock:
                try:
                    sock.close()
                except:
                    pass
            self.running = False
    
    def run(self):
        """运行服务（24小时在线被动监听）"""
        logger.info("=" * 70)
        logger.info("MT5数据服务 - TCP Socket高速模式")
        logger.info("=" * 70)
        logger.info("")
        logger.info("模式: 24小时在线被动监听")
        logger.info(f"监听地址: {self.host}:{self.port}")
        logger.info("传输格式: 二进制（40字节/条，最快速度）")
        logger.info("不存储任何数据，仅作为MT5数据封装")
        logger.info("")
        logger.info("等待Linux服务器连接...")
        logger.info("")
        logger.info("=" * 70)
        logger.info("")
        
        # 24小时在线，自动重连
        while True:
            try:
                self.listen_for_clients()
            except KeyboardInterrupt:
                logger.info("\n服务停止")
                self.running = False
                break
            except Exception as e:
                logger.error(f"\n服务异常: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                logger.info("5秒后自动重启监听...")
                time.sleep(5)
                self.running = False


if __name__ == '__main__':
    import sys
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(BASE_DIR))
    sys.path.insert(0, str(BASE_DIR / "src"))
    
    from loguru import logger
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.remove()
    logger.add(log_dir / "mt5_service_{time}.log", rotation="10 MB", retention="7 days", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    
    service = MT5DataService(host="0.0.0.0", port=8888)
    service.run()
