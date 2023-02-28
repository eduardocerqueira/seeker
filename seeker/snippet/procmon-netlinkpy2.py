#date: 2023-02-28T17:05:42Z
#url: https://api.github.com/gists/366757893156aa1e101e82e370fde44a
#owner: https://api.github.com/users/nikaiw

#!/usr/bin/env python2

import socket
import os
import struct

if getattr(socket, "NETLINK_CONNECTOR", None) is None:
    socket.NETLINK_CONNECTOR = 11

CN_IDX_PROC = 1
CN_VAL_PROC = 1

NLMSG_NOOP = 1
NLMSG_ERROR = 2
NLMSG_DONE = 3
NLMSG_OVERRUN = 4

PROC_CN_MCAST_LISTEN = 1
PROC_CN_MCAST_IGNORE = 2

PROC_EVENT_WHAT = {
    0x00000000: "PROC_EVENT_NONE",
    0x00000001: "PROC_EVENT_FORK",
    0x00000002: "PROC_EVENT_EXEC",
    0x00000004: "PROC_EVENT_UID",
    0x00000040: "PROC_EVENT_GID",
    0x00000080: "PROC_EVENT_SID",
    0x00000100: "PROC_EVENT_PTRACE",
    0x00000200: "PROC_EVENT_COMM",
    0x00000400: "PROC_EVENT_COREDUMP",
    0x80000000: "PROC_EVENT_EXIT",
}
PROC_EVENT_NONE = 0

def get_process_name(pid):
    try:
        with open("/proc/%s/cmdline" % pid, "rb") as f:
            cmdline = f.read().decode().rstrip("\0")
            return cmdline.split("\0")[0]
    except IOError:
        return ""

def main():
    import platform
    assert platform.processor() == "x86_64"

    # Create Netlink socket
    sock = socket.socket(socket.AF_NETLINK, socket.SOCK_DGRAM,
                         socket.NETLINK_CONNECTOR)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
    sock.bind((os.getpid(), CN_IDX_PROC))

    # Send PROC_CN_MCAST_LISTEN
    data = struct.pack("=IHHII IIIIHH I",
                      16 + 20 + 4, NLMSG_DONE, 0, 0, os.getpid(),
                      CN_IDX_PROC, CN_VAL_PROC, 0, 0, 4, 0,
                      PROC_CN_MCAST_LISTEN)
    if sock.send(data) != len(data):
        raise RuntimeError("Failed to send PROC_CN_MCAST_LISTEN")

    while True:
        data, (nlpid, nlgrps) = sock.recvfrom(1024)

        # Netlink message header (struct nlmsghdr)
        msg_len, msg_type, msg_flags, msg_seq, msg_pid \
            = struct.unpack("=IHHII", data[:16])
        data = data[16:]

        if msg_type == NLMSG_NOOP:
            continue
        if msg_type in (NLMSG_ERROR, NLMSG_OVERRUN):
            break

        # Connector message header (struct cn_msg)
        cn_idx, cn_val, cn_seq, cn_ack, cn_len, cn_flags \
            = struct.unpack("=IIIIHH", data[:20])
        data = data[20:]

        # Process event message (struct proc_event)
        what, cpu, timestamp = struct.unpack("=LLQ", data[:16])
        data = data[16:]
        if what != PROC_EVENT_NONE:
            # I'm just extracting PID (meaning varies) for example purposes
            pid = struct.unpack("=I", data[:4])[0]
            what = PROC_EVENT_WHAT.get(what, "PROC_EVENT_UNKNOWN(%d)" % what)
            process_name = get_process_name(pid)
            if process_name:
                print("%s: CPU%d PID=%d Name=%s" % (what, cpu, pid, process_name))
            else:
                print("%s: CPU%d PID=%d Name=unknown (not found)" % (what, cpu, pid))

if __name__ == "__main__":
    main()