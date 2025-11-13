#date: 2025-11-13T17:02:49Z
#url: https://api.github.com/gists/f99a8104b419a7b2f6437a18ce2a8790
#owner: https://api.github.com/users/wangwei334455

"""
数据接收服务 - 通过 gRPC StreamTicks 被动接收Windows前置机推送的TICK数据

架构说明：
- gRPC StreamTicks 是服务器端流（server-side streaming）
- 客户端建立连接后，服务器持续推送TICK数据
- 客户端被动接收数据流（不是主动拉取）
- 数据存储到Redis，供L2策略核心消费

统一架构: gRPC StreamTicks - 类型安全、性能优异、统一协议
"""
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SRC_DIR))

import redis
from loguru import logger
from config.redis_config import REDIS_CONFIG, REDIS_KEYS


class DataPuller:
    """
    从Windows前置机通过 gRPC StreamTicks 被动接收TICK数据
    
    工作原理：
    - 客户端建立gRPC连接（主动）
    - 服务器持续推送TICK数据流（被动接收）
    - 数据存储到Redis，供L2策略核心消费
    
    统一使用 gRPC，无降级方案
    """
    
    def __init__(
        self, 
        frontend_host="192.168.10.131", 
        grpc_port=50051,
        symbol="BTCUSDm"
    ):
        self.frontend_host = frontend_host
        self.grpc_port = grpc_port
        self.symbol = symbol
        self.redis_client = redis.Redis(**REDIS_CONFIG)
        
        # gRPC 相关
        self.grpc_client = None
        self.grpc_available = False
        
        # 🔴 关键：TICK序列号生成器（从1开始，供L2策略核心顺序检查）
        self.tick_seq = 0  # 下一个TICK的序列号（从1开始）
        
        logger.info(f"数据接收服务初始化 - gRPC StreamTicks (服务器推送)")
        logger.info(f"前置机: {frontend_host}:{grpc_port}")
        logger.info(f"交易品种: {symbol}")
        logger.info(f"工作模式: 被动接收服务器推送的TICK数据流")
    
    def _init_grpc_client(self) -> bool:
        """初始化 gRPC 客户端"""
        try:
            from src.trading.services.grpc_trade_client import get_grpc_client, is_grpc_available
            
            if not is_grpc_available():
                logger.error("gRPC 不可用，请先运行: python scripts/generate_grpc_code.py --target linux")
                return False
            
            self.grpc_client = get_grpc_client(host=self.frontend_host, port=self.grpc_port)
            self.grpc_client._ensure_connected()
            self.grpc_available = True
            logger.info(f"✅ gRPC 客户端初始化成功: {self.frontend_host}:{self.grpc_port}")
            return True
        except Exception as e:
            logger.error(f"❌ gRPC 客户端初始化失败: {e}")
            logger.error("请确保:")
            logger.error("1. Windows MT5 中继服务已启动 gRPC 服务（端口 50051）")
            logger.error("2. gRPC 代码已生成: python scripts/generate_grpc_code.py --target linux")
            self.grpc_available = False
            return False
    
    def save_tick_to_redis(self, tick_data: dict):
        """
        保存TICK数据到Redis（使用pipeline批量写入 + Pub/Sub 通知）
        
        Redis 三向存储 + Pub/Sub 通知:
        1. Sorted Set - 历史查询
        2. Stream - 实时流（供L2消费）
        3. String - 最新快照（O(1)查询）
        4. Pub/Sub - 实时通知（供API Server订阅）
        
        🔴 关键：为每个TICK添加seq和checksum，供L2策略核心顺序检查
        """
        try:
            import json
            import hashlib
            
            # 🔴 关键：生成序列号和校验和（供L2策略核心顺序检查）
            self.tick_seq += 1
            tick_data['seq'] = self.tick_seq
            
            # 生成校验和（防止数据篡改）
            checksum_base = f"{tick_data.get('time_msc', 0)}:{self.tick_seq}:{tick_data.get('bid', 0)}:{tick_data.get('ask', 0)}"
            tick_data['checksum'] = hashlib.md5(checksum_base.encode('utf-8')).hexdigest()[:8]
            
            tick_json = json.dumps(tick_data, ensure_ascii=False)
            tick_time_msc = tick_data.get('time_msc', 0)
            
            tick_data_key = REDIS_KEYS['tick_data']
            tick_stream_key = REDIS_KEYS['tick_stream']
            latest_tick_key = REDIS_KEYS['latest_tick']
            
            pipe = self.redis_client.pipeline()
            
            # 1. 写入Sorted Set（历史查询）
            pipe.zadd(tick_data_key, {tick_json: tick_time_msc})
            
            # 2. 写入Stream（实时流，供L2消费）
            pipe.xadd(tick_stream_key, {'value': tick_json}, id='*', maxlen=1000, approximate=True)
            
            # 3. 更新最新TICK（快照，O(1)查询）
            pipe.set(latest_tick_key, tick_json)
            
            # 执行批量写入
            pipe.execute()
            
            # 4. 发布 Pub/Sub 通知（实时通知，供API Server订阅）
            try:
                self.redis_client.publish(f"tick:{self.symbol}", "new_tick")
            except Exception as e:
                logger.debug(f"Pub/Sub 通知失败（非关键）: {e}")
            
        except Exception as e:
            logger.error(f"保存TICK到Redis失败: {e}")
    
    def run(self, reconnect_interval=5):
        """
        运行数据接收服务
        
        工作模式：被动接收服务器推送的TICK数据流
        - 客户端建立gRPC连接（主动）
        - 服务器持续推送TICK数据（被动接收）
        - 数据存储到Redis
        
        统一使用 gRPC StreamTicks，无降级方案
        """
        logger.info("=" * 70)
        logger.info("数据接收服务启动 - gRPC StreamTicks (服务器推送)")
        logger.info("=" * 70)
        logger.info(f"前置机: {self.frontend_host}:{self.grpc_port}")
        logger.info(f"交易品种: {self.symbol}")
        logger.info(f"工作模式: 被动接收服务器推送的TICK数据流")
        logger.info("")
        
        # 初始化 gRPC 连接
        if not self._init_grpc_client():
            logger.error("❌ gRPC 初始化失败，服务退出")
            return
        
        try:
            while True:
                if not self.grpc_available:
                    logger.warning("gRPC 连接断开，尝试重连...")
                    if not self._init_grpc_client():
                        logger.error(f"gRPC 重连失败，{reconnect_interval}秒后重试...")
                        time.sleep(reconnect_interval)
                        continue
                
                try:
                    # 被动接收 gRPC 流式数据（服务器推送，阻塞调用）
                    # 服务器持续推送TICK数据，客户端通过迭代器被动接收
                    for tick_data in self.grpc_client.stream_ticks(self.symbol):
                        # 确保数据格式正确
                        if tick_data and 'time_msc' in tick_data:
                            # 添加 symbol 字段（如果缺失）
                            if 'symbol' not in tick_data:
                                tick_data['symbol'] = self.symbol
                            self.save_tick_to_redis(tick_data)
                        else:
                            logger.warning(f"收到无效TICK数据: {tick_data}")
                    
                    # 流结束（不应该发生，除非服务端关闭）
                    logger.warning("gRPC StreamTicks 流结束，尝试重连...")
                    self.grpc_available = False
                    
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"gRPC 拉取数据异常: {e}")
                    self.grpc_available = False
                    logger.warning(f"等待 {reconnect_interval} 秒后重连...")
                    time.sleep(reconnect_interval)
                
        except KeyboardInterrupt:
            logger.info("\n数据拉取服务停止")
            self.cleanup()
        except Exception as e:
            logger.error(f"拉取服务异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        if self.grpc_client:
            try:
                self.grpc_client.close()
            except:
                pass


if __name__ == '__main__':
    puller = DataPuller(
        frontend_host="192.168.10.131",
        grpc_port=50051,
        symbol="BTCUSDm"
    )
    puller.run(reconnect_interval=5)
