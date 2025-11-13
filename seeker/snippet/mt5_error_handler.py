#date: 2025-11-13T17:03:12Z
#url: https://api.github.com/gists/d68fdd03a1f1096971a5f4ed71530eb2
#owner: https://api.github.com/users/wangwei334455

"""
MT5错误处理器 - 详细解析错误代码并提供解决方案
"""
from typing import Tuple, Dict
import MetaTrader5 as mt5


# MT5错误代码映射表（基于官方文档）
MT5_ERROR_CODES: Dict[int, Tuple[str, str]] = {
    # 通用错误
    -1: ("通用错误", "检查MT5终端是否正常运行"),
    -2: ("无效参数", "检查传递的参数是否正确"),
    -3: ("内存不足", "重启MT5终端或增加系统内存"),
    -4: ("没有历史数据", "检查MT5终端是否下载了该品种的历史数据"),
    -5: ("历史数据不足", "在MT5中手动下载更多历史数据"),
    -6: ("认证失败", "检查账号、密码、服务器是否正确"),
    -7: ("未知交易品种", "检查交易品种名称是否正确"),
    -8: ("无效价格", "检查价格是否在合理范围内"),
    -9: ("无效订单", "检查订单参数是否正确"),
    -10: ("交易被禁用", "检查账户是否允许交易"),
    
    # 连接相关错误
    -10001: ("网络错误", "检查网络连接"),
    -10002: ("连接失败", "检查MT5服务器地址和端口"),
    -10003: ("连接超时", "增加timeout参数或检查网络"),
    -10004: ("请求超时", "增加timeout参数"),
    -10005: ("连接被拒绝", "检查防火墙设置"),
    -10006: ("连接已断开", "重新连接MT5"),
    -10007: ("市场关闭", "等待市场开盘"),
    
    # 交易相关错误
    -10010: ("订单已过期", "使用更长的订单有效期"),
    -10011: ("订单价格错误", "检查订单价格是否合理"),
    -10012: ("订单手数错误", "检查订单手数是否符合要求"),
    -10013: ("订单止损错误", "检查止损价格是否合理"),
    -10014: ("订单止盈错误", "检查止盈价格是否合理"),
    -10015: ("交易品种不可用", "检查该品种是否可交易"),
    -10016: ("账户余额不足", "增加账户余额"),
    -10017: ("仓位已满", "平仓后再开新仓"),
    -10018: ("订单冻结", "等待订单解冻"),
    -10019: ("无效操作", "检查操作是否被允许"),
    -10020: ("订单不存在", "检查订单号是否正确"),
    
    # 数据相关错误
    -10100: ("数据格式错误", "检查数据格式是否正确"),
    -10101: ("数据过期", "获取最新数据"),
    -10102: ("数据不完整", "重新获取数据"),
    -10103: ("数据校验失败", "检查数据完整性"),
    
    # 权限相关错误
    -10200: ("权限不足", "检查账户权限"),
    -10201: ("只读模式", "切换到可写模式"),
    -10202: ("功能被禁用", "启用该功能"),
    
    # 其他错误
    0: ("成功", "操作成功完成"),
    1: ("成功", "操作成功完成"),
}


class MT5ErrorHandler:
    """MT5错误处理器"""
    
    @staticmethod
    def get_last_error() -> Tuple[int, str, str, str]:
        """
        获取最后一次MT5错误的详细信息
        
        Returns:
            Tuple[int, str, str, str]: (错误代码, 原始错误消息, 错误名称, 解决方案)
        """
        error_code, error_msg = mt5.last_error()
        
        if error_code in MT5_ERROR_CODES:
            error_name, solution = MT5_ERROR_CODES[error_code]
        else:
            error_name = "未知错误"
            solution = f"请查阅MT5官方文档或联系技术支持（错误代码: {error_code}）"
        
        return error_code, error_msg, error_name, solution
    
    @staticmethod
    def format_error(error_code: int, error_msg: str = None) -> str:
        """
        格式化错误信息
        
        Args:
            error_code: 错误代码
            error_msg: 原始错误消息
            
        Returns:
            str: 格式化后的错误信息
        """
        if error_code in MT5_ERROR_CODES:
            error_name, solution = MT5_ERROR_CODES[error_code]
        else:
            error_name = "未知错误"
            solution = "请查阅MT5官方文档"
        
        msg_parts = [
            f"MT5错误 [{error_code}]: {error_name}"
        ]
        
        if error_msg:
            msg_parts.append(f"原始消息: {error_msg}")
        
        msg_parts.append(f"解决方案: {solution}")
        
        return "\n".join(msg_parts)
    
    @staticmethod
    def log_error(logger, context: str = ""):
        """
        记录MT5错误到日志
        
        Args:
            logger: 日志记录器
            context: 错误上下文（如"初始化MT5时"）
        """
        error_code, error_msg, error_name, solution = MT5ErrorHandler.get_last_error()
        
        if context:
            logger.error(f"{context} - {MT5ErrorHandler.format_error(error_code, error_msg)}")
        else:
            logger.error(MT5ErrorHandler.format_error(error_code, error_msg))
    
    @staticmethod
    def is_critical_error(error_code: int) -> bool:
        """
        判断是否为严重错误（需要立即处理）
        
        Args:
            error_code: 错误代码
            
        Returns:
            bool: 是否为严重错误
        """
        critical_errors = {
            -6,      # 认证失败
            -10001,  # 网络错误
            -10002,  # 连接失败
            -10003,  # 连接超时
            -10006,  # 连接已断开
        }
        return error_code in critical_errors
    
    @staticmethod
    def is_retryable_error(error_code: int) -> bool:
        """
        判断错误是否可重试
        
        Args:
            error_code: 错误代码
            
        Returns:
            bool: 是否可重试
        """
        retryable_errors = {
            -10001,  # 网络错误
            -10003,  # 连接超时
            -10004,  # 请求超时
            -10006,  # 连接已断开
        }
        return error_code in retryable_errors


# 便捷函数
def get_last_error_info() -> Tuple[int, str, str, str]:
    """获取最后一次MT5错误的详细信息"""
    return MT5ErrorHandler.get_last_error()


def format_mt5_error(error_code: int, error_msg: str = None) -> str:
    """格式化MT5错误信息"""
    return MT5ErrorHandler.format_error(error_code, error_msg)


def log_mt5_error(logger, context: str = ""):
    """记录MT5错误到日志"""
    MT5ErrorHandler.log_error(logger, context)

