#date: 2022-07-04T02:46:11Z
#url: https://api.github.com/gists/c643d0bf4da8b5c57f74a35fbaeadce0
#owner: https://api.github.com/users/kaiili

from psutil import Process, net_connections
from requests import get
from rich import print


def check_port_ishttp(port: int) -> bool:
    """通过输入的端口检查本地端口是否为 http协议
    """
    url = f"http://127.0.0.1:{port}"
    try:
        get(url, timeout=0.5)
        return True
    except:
        return False


def get_ports() -> list:
    """获取本机监听的所有端口
    """
    pass


def get_proccessname_by_pid(pid: str) -> str:
    """通过 pid换 proccess name , proccess 路径
    """
    return Process(pid=pid).name()


def check_json(port: int) -> bool:

    url = f"http://127.0.0.1:{port}/json"
    try:
        code = get(url, timeout=0.5).status_code
        if code == 200:
            return True
        else:
            return False
    except:
        return False


def test():
    need_status = "LISTEN"
    conn_list = net_connections(kind="tcp")
    for conn in conn_list:
        fd, family, type, laddr, raddr, status, pid = conn
        if status != need_status:
            continue
        if check_port_ishttp(laddr[1]) is False:
            continue

        if check_json(laddr[1]) is True:
            print(laddr[1], pid, get_proccessname_by_pid(pid))


if __name__ == "__main__":
    test()
