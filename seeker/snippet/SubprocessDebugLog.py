#date: 2023-08-09T17:06:34Z
#url: https://api.github.com/gists/9648d43a41ffd55a1909a4ecedef99b4
#owner: https://api.github.com/users/jnj16180340

'''
USAGE:

subprocess_debuglog(
    ["./arglog",'arg1','--arg2', '-arg',3],
    text=True,
    encoding='utf-8',
)
'''

import subprocess
import time
import contextlib

#SEE https://gist.github.com/thelinuxkid/5114777
newlines = ['\n', '\r\n', '\r']
def unbuffered(proc, stream='stdout'):
    stream = getattr(proc, stream)
    with contextlib.closing(stream):
        while True:
            out = []
            last = stream.read(1)
            # Don't loop forever
            if last == '' and proc.poll() is not None:
                break
            while last not in newlines:
                # Don't loop forever
                if last == '' and proc.poll() is not None:
                    break
                out.append(last)
                last = stream.read(1)
            out = ''.join(out)
            yield out

def subprocess_debuglog(command: list[str], *subprocess_args, **subprocess_kwargs):
    TSTART = time.perf_counter()
    # Due to io buffering, Popen.communicate() blocks so it cant be used in real time
    # alternative ?is? calling sys.stdout.flush() in a while loop or sth
    process = subprocess.Popen(
        [str(s) for s in command],
        *subprocess_args,
        **subprocess_kwargs,
        # Necessary for .communicate() to not return null
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    for line in unbuffered(process):
        #print line
        print(f"[t = {time.perf_counter() - TSTART}s] PYTHON SAYS: STDIO LINE is: {line}")
