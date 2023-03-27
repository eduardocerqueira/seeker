#date: 2023-03-27T16:39:41Z
#url: https://api.github.com/gists/d72c2c7fedacadb14f67bd909658bfbc
#owner: https://api.github.com/users/jdurgin

#!/usr/bin/python3

from datetime import datetime
import re
import sys

class Transaction(object):
    def __init__(self, tid, start):
        self.tid = tid
        self.start = start
    def set_end(self, end):
        self.end = end
        td = self.end - self.start
        self.duration = (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / float(10**6)

def main():
    in_flight = {}
    durations = []
    time_format = '%Y-%m-%dT%H:%M:%S.%f%z'
    with open(sys.argv[1], 'r') as f:
        for line in f.readlines():
            if '-->' in line and 'osd_op(' in line:
                try:
                    tid = re.search('osd_op\([^:]+:(\d+)', line).group(1)
                except Exception:
                    print('bad line:', line)
                    raise
                start = datetime.strptime(line[:28], time_format)
                if tid in in_flight:
                    print('resent', tid)
                else:
                    in_flight[tid] = Transaction(tid, start)
            elif '<==' in line and 'osd_op_reply(' in line:
                tid = re.search('osd_op_reply\((\d+)', line).group(1)
                if tid in in_flight:
                    trans = in_flight[tid]
                    trans.set_end(datetime.strptime(line[:28], time_format))
                    durations.append((trans.tid, trans.duration))
                    del in_flight[tid]
                else:
                    print('dup reply', tid)
    for tid in in_flight:
        print('unacked request', tid)
    durations.sort(key=lambda x: x[1], reverse=True)
    print()
    print('100 longest requests:')
    for i in range(min(100, len(durations))):
        print(i + 1, durations[i][1], durations[i][0])

if __name__ == '__main__':
    main()