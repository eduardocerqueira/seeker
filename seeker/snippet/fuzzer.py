#date: 2022-10-05T17:35:56Z
#url: https://api.github.com/gists/c71ad62f7b3a681dd8b0943e1da8e14f
#owner: https://api.github.com/users/OlivierLaflamme

from boofuzz import *

IP = "192.168.0.5"
PORT = 80

def check_response(target, fuzz_data_logger, session, *args, **kwargs):
    fuzz_data_logger.log_info("Checking test case response...")
    try:
        response = target.recv(512)
    except:
        fuzz_data_logger.log_fail("Unable to connect to target. Closing...")
        target.close()
        return

    #if empty response
    if not response:
        fuzz_data_logger.log_fail("Empty response, target may be hung. Closing...")
        target.close()
        return

    #remove everything after null terminator, and convert to string
    #response = response[:response.index(0)].decode('utf-8')
    fuzz_data_logger.log_info("response check...\n" + response.decode())
    target.close()
    return
    
def main():
    '''
    options = {
        "start_commands": [
            "sudo chroot /home/boschko/Documents/firmware/TTD.bin.extracted/squashfs-root ./httpd"
        ],
        "stop_commands": ["echo stopping"],
        "proc_name": ["/usr/bin/qemu-arm-static ./httpd"]
    }
    procmon = ProcessMonitor("127.0.0.1", 26002)
    procmon.set_options(**options)
    '''

    session = Session(
        target=Target(
            connection=SocketConnection(IP, PORT, proto="tcp"),
            # monitors=[procmon]
        ),
        post_test_case_callbacks=[check_response],
    )

    s_initialize(name="Request")
    with s_block("Request-Line"):
        # Line 1
        s_group("Method", ["GET"])
        s_delim(" ", fuzzable=False, name="space-1-1")
        s_string("/goform/123", fuzzable=False)    # fuzzable 1
        s_delim(" ", fuzzable=False, name="space-1-2")
        s_static("HTTP/1.1", name="HTTP_VERSION")
        s_static("\r\n", name="Request-Line-CRLF-1")
        # Line 2
        s_static("Host")
        s_delim(": ", fuzzable=False, name="space-2-1")
        s_string("192.168.0.5", fuzzable=False, name="IP address")
        s_static("\r\n", name="Request-Line-CRLF-2")
        # Line 3
        s_static("Connection")
        s_delim(": ", fuzzable=False, name="space-3-1")
        s_string("keep-alive", fuzzable=False, name="Connection state")
        s_static("\r\n", name="Request-Line-CRLF-3")
        # Line 4
        s_static("Cookie")
        s_delim(": ", fuzzable=False, name="space-4-1")
        s_string("bLanguage", fuzzable=False, name="key-bLanguage")
        s_delim("=", fuzzable=False)
        s_string("en", fuzzable=False, name="value-bLanguage")
        s_delim("; ", fuzzable=False)
        s_string("password", fuzzable= "**********"="key-password")
        s_delim("=", fuzzable=False)
        s_string("ce24124987jfjekfjlasfdjmeiruw398r", fuzzable=True)    # fuzzable 2
        s_static("\r\n", name="Request-Line-CRLF-4")
        # over
        s_static("\r\n")
        s_static("\r\n")

    session.connect(s_get("Request"))
    session.fuzz()

if __name__ == "__main__":
    main()

