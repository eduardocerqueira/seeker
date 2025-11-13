#date: 2025-11-13T17:02:54Z
#url: https://api.github.com/gists/4c793df7bd6fb44895f49714b67dd982
#owner: https://api.github.com/users/wangwei334455

"""
环境检查和交易安全模块

🔴 核心安全机制：防止测试订单在生产环境意外执行

使用方式：
1. 设置环境变量：export TRADING_ENV=LIVE  # 生产环境
2. 或：export TRADING_ENV=TEST  # 测试环境（默认）
3. 在订单执行前调用 is_production_mode() 检查
"""
import os
from loguru import logger
from typing import Optional

# 环境变量名称
TRADING_ENV_VAR = 'TRADING_ENV'

# 环境值定义
ENV_PRODUCTION = 'LIVE'  # 生产环境（允许真实交易）
ENV_TEST = 'TEST'        # 测试环境（禁止真实交易）
ENV_DEMO = 'DEMO'        # 模拟环境（禁止真实交易）

# 默认环境（安全起见，默认为测试环境）
DEFAULT_ENV = ENV_TEST


def get_trading_env() -> str:
    """
    获取当前交易环境
    
    Returns:
        str: 环境值 ('LIVE', 'TEST', 'DEMO')
    """
    env = os.environ.get(TRADING_ENV_VAR, DEFAULT_ENV).upper()
    
    # 验证环境值有效性
    valid_envs = [ENV_PRODUCTION, ENV_TEST, ENV_DEMO]
    if env not in valid_envs:
        logger.warning(
            f"⚠️ 无效的 {TRADING_ENV_VAR} 值: {env}, "
            f"使用默认值: {DEFAULT_ENV} (测试环境)"
        )
        env = DEFAULT_ENV
    
    return env


def is_production_mode() -> bool:
    """
    检查当前是否为生产模式（允许真实交易）
    
    🔴 安全机制：只有明确设置为 LIVE 时才允许执行真实订单
    
    Returns:
        bool: True 表示生产模式，False 表示测试/模拟模式
    """
    env = get_trading_env()
    is_prod = env == ENV_PRODUCTION
    
    if not is_prod:
        logger.debug(f"当前环境: {env} (非生产模式，禁止真实交易)")
    
    return is_prod


def require_production_mode(func_name: str = "执行交易") -> bool:
    """
    要求生产模式，如果不是则抛出异常
    
    Args:
        func_name: 函数名称（用于错误提示）
        
    Returns:
        bool: 如果是生产模式返回 True
        
    Raises:
        EnvironmentError: 如果不是生产模式
    """
    if not is_production_mode():
        env = get_trading_env()
        error_msg = (
            f"🚫 安全阻止: 尝试在非生产环境 ({env}) 中 {func_name}！\n"
            f"   当前环境: {env}\n"
            f"   要启用真实交易，请设置环境变量: {TRADING_ENV_VAR}={ENV_PRODUCTION}\n"
            f"   例如: export {TRADING_ENV_VAR}={ENV_PRODUCTION} && python api_server.py"
        )
        logger.error(error_msg)
        raise EnvironmentError(error_msg)
    
    return True


def get_env_info() -> dict:
    """
    获取环境信息（用于日志和调试）
    
    Returns:
        dict: 环境信息字典
    """
    env = get_trading_env()
    return {
        'env': env,
        'is_production': is_production_mode(),
        'env_var': TRADING_ENV_VAR,
        'env_value': os.environ.get(TRADING_ENV_VAR, f'未设置（默认: {DEFAULT_ENV}）'),
    }


def log_env_status():
    """记录当前环境状态（启动时调用）"""
    info = get_env_info()
    env = info['env']
    is_prod = info['is_production']
    
    if is_prod:
        logger.warning("=" * 70)
        logger.warning("⚠️  生产模式已启用 - 真实交易将被执行！")
        logger.warning("=" * 70)
        logger.warning(f"   环境变量: {TRADING_ENV_VAR}={env}")
        logger.warning("   请确保：")
        logger.warning("   1. 已充分测试策略")
        logger.warning("   2. 已设置合理的风险控制")
        logger.warning("   3. 已监控系统运行状态")
        logger.warning("=" * 70)
    else:
        logger.info(f"✅ 测试模式: {env} (真实交易已禁用)")
        logger.info(f"   要启用生产模式，请设置: export {TRADING_ENV_VAR}={ENV_PRODUCTION}")

