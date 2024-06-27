//date: 2024-06-27T16:53:15Z
//url: https://api.github.com/gists/341ea3bc0287a02a19bcd89ee021f9a0
//owner: https://api.github.com/users/theoremoon

import random
import time
from typing import Deque
from ptrlib import Socket
from itertools import combinations
from collections import Counter, deque

def _mul(x, y):
    if x == 0:
        x = 2**16
    if y == 0:
        y = 2**16
    z = x * y % (2**16 + 1)
    return z % 2**16

def _add(x, y):
    return (x + y) & 0xffff

def _sub(x, y):
    return (x - y) & 0xffff

def _inv(x):
    if x == 0:
        return pow(2**16, -1, 2**16 + 1)
    return pow(x, -1, 2**16 + 1)

def decompose(v):
    x1 = (v >> 48) & 0xffff
    x2 = (v >> 32) & 0xffff
    x3 = (v >> 16) & 0xffff
    x4 = v & 0xffff
    return x1, x2, x3, x4

def search_k18(pairs, pairs2):
    k18_cands = []
    for k18 in range(2**16):
        k18inv = _inv(k18)
        flag = True
        for i in range(N):
            _, ct = pairs[i]
            _, ct_ = pairs2[i]
            y1, _, y3, _ = decompose(ct)
            y1_, _, y3_, _ = decompose(ct_)
            tdiff = _mul(y1, k18inv) ^ _mul(y1_, k18inv)
            tdiff2 = y3 ^ y3_
            if tdiff & 1 != tdiff2 & 1:
                flag = False
                break
        if flag:
            k18_cands.append(k18)
    return k18_cands

def search_k20(pairs, pairs2, k18):
    k18inv = _inv(k18)
    candidates = []
    for k20 in range(2**16):
        flag = True
        for i in range(N):
            _, ct = pairs[i]
            _, ct_ = pairs2[i]
            y1, _, y3, _ = decompose(ct)
            y1_, _, y3_, _ = decompose(ct_)
            tdiff = _mul(y1, k18inv) ^ _mul(y1_, k18inv)
            tdiff2 = _sub(y3, k20) ^ _sub(y3_, k20)
            if tdiff != tdiff2:
                flag = False
                break
        if flag:
            candidates.append(k20)
    return candidates

def check_is_good_pair(pair, pair3, k18inv, k20):
    _, ct = pair
    _, ct_ = pair3
    y1, _, y3, _ = decompose(ct)
    y1_, _, y3_, _ = decompose(ct_)

    ydiff = _mul(y1, k18inv) ^ _mul(y1_, k18inv)
    ydiff2 = _sub(y3, k20) ^ _sub(y3_, k20 ^ 2)   # 127のときのk20は差分があるので考慮
    # return (ydiff ^ ydiff2) in [0x100, 0x300, 0x700, 0xf00, 0x1f00, 0x3f00, 0x7f00, 0xff00]
    return (ydiff ^ ydiff2) in [0x100, 0x300]

def search_k21(good_pairs, k18, k20):
    candidates = set()
    k19_cands = set()

    # good pairs から2個選んでくる。その2個が真にgoodであることを祈りながら
    for comb in combinations(good_pairs, 2):
        table = {}
        for k21 in range(2**16):
            k21inv = _inv(k21)

            udiffs = []
            for _, ct, _, ct_ in comb:
                _, _, _, y4 = decompose(ct)
                _, _, _, y4_ = decompose(ct_)
                udiff = _mul(y4, k21inv) ^ _mul(y4_, k21inv)
                udiffs.append(udiff)
            key = tuple(udiffs)
            table[key] = k21

        for k19 in range(2**16):
            udiffs = []
            for _, ct, _, ct_ in comb:
                _, y2, _, _ = decompose(ct)
                _, y2_, _, _ = decompose(ct_)
                udiff = _sub(y2, k19) ^ _sub(y2_, k19)
                udiffs.append(udiff)
            key = tuple(udiffs)
            if key in table:
                k21 = table[key]
                candidates.add(k21)
                k19_cands.add(k19)
    return candidates

def search_k19(pairs, k21):
    k21inv = _inv(k21)

    counter = Counter()
    for k19 in range(2**16):
        for _, ct, _, ct_ in pairs:
            _, y2, _, y4 = decompose(ct)
            _, y2_, _, y4_ = decompose(ct_)
            udiff = _mul(y4, k21inv) ^ _mul(y4_, k21inv)
            udiff2 = _sub(y2, k19) ^ _sub(y2_, k19)
            if udiff == udiff2:
                counter[k19] += 1

    common = counter.most_common()
    res = []
    for k, v in common:
        if v == common[0][1]:
            res.append(k)
        else:
            break
    return res

q: Deque[int] = deque()

sock: Socket = None
def reconnect():
    global sock, start
    if sock is not None: sock.close()
    start = time.time()
    q.clear()
    sock = Socket("idea.2024.ctfcompetition.com", 1337)
    import socket
    sock._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)

def elapsed(msg):
    print(msg, 'elapsed:', time.time() - start)

while True:
    try:
        print("\n\nstart new search")
        reconnect()

        def encrypt_send(pt: int):
            sock.sendline("1")
            sock.sendline(hex(pt))
            q.append(pt)

        def encrypt_recv():
            sock.recvuntil("text: ")
            pt = q.popleft()
            return pt, int(sock.recvline().strip())

        def mask(mask: int):
            sock.sendline("2")
            sock.sendlineafter("mask: ", hex(mask))


        # step1. find k18, k20
        N = 20   # step1で使う数
        M = 400  # step2で使う数

        base = random.randrange(0, 2**48)
        pairs = []
        for i in range(M):
            pt = (random.randrange(0, 2**16) << 48)  | base
            encrypt_send(pt)

        for i in range(M):
            pairs.append(encrypt_recv())

        elapsed("recv first wave")

        mask(2**103)
        pairs2 = []
        for i in range(N):
            pt, ct = pairs[i]
            pt_ = pt + 2**(16 + 16 + 7)
            encrypt_send(pt_)
        for i in range(N):
            pairs2.append(encrypt_recv())

        elapsed("recv second wave")

        k18_cands = search_k18(pairs, pairs2)
        elapsed("k18 found")
        if len(k18_cands) == 0:
            # - のパターンをやる
            for i in range(N):
                pt, ct = pairs[i]
                pt_ = pt - 2**(16 + 16 + 7)
                ct_ = encrypt_send(pt_)
            for i in range(N):
                pairs2[i] = encrypt_recv()
            k18_cands = search_k18(pairs, pairs2)
            assert len(k18_cands) > 0, "k18 not found"

        for k18 in k18_cands:
            k20_cands = search_k20(pairs, pairs2, k18)
            if len(k20_cands) > 0:
                k20 = k20_cands[1]  # 決め打ち
                break
        assert len(k20_cands) > 0, "k20 not found"
        elapsed("k20 found")
        print("k18:", k18)
        print("k20:", k20_cands)

        # step2. find k21, k19
        mask((2**103) ^ (2**127))
        good_pairs = []  # k0 のdiff消せてるペア。偽陽性がある
        k18inv = _inv(k18)

        for _ in range(M):
            pt_ = (random.randrange(0, 2**16) << 48) | base
            encrypt_send(pt_)

        for _ in range(M):
            pt_, ct_ = encrypt_recv()
            for i in range(M):
                pt, ct = pairs[i]
                if check_is_good_pair((pt, ct), (pt_, ct_), k18inv, k20):
                    good_pairs.append([pt, ct, pt_, ct_])
                    # あるptに対してgoodなpt_は1つだけしかないのでbreakしてよい
                    break
            if len(good_pairs) >= 10:
                break

        q.clear() # break するので
        assert len(good_pairs) >= 4, "good_pairs not found"
        elapsed("good pair found")

        k21_cands = search_k21(good_pairs[:5], k18, k20)
        assert len(k21_cands) > 0, "k21 not found"
        print("k21:", k21_cands)
        elapsed("k21 found")

        pairs_for_k16 = []
        for i in range(N):
            _, ct = pairs[i]
            _, ct_ = pairs2[i]

            y1, y2, y3, y4 = decompose(ct)
            y1_, y2_, y3_, y4_ = decompose(ct_)
            pairs_for_k16.append([[y1, y2, y3, y4], [y1_, y2_, y3_, y4_]])

        import json

        args = []

        for k21 in k21_cands:
            k19_cands = search_k19(good_pairs, k21)
            print("k19:", k19_cands)

            for k19 in k19_cands:
                args.append(json.dumps({
                    "pairs": pairs_for_k16,
                    "k18": k18,
                    "k19": k19,
                    "k20": k20,
                    "k21": k21,
                }))

        print('len:', len(args))
        elapsed("start processing")

        import subprocess
        import concurrent.futures
        from multiprocessing import Pool

        def solve_k16k17(arg):
            output = subprocess.run(["./go_solver/k16k17", arg], capture_output=True)
            k16k17 = output.stdout.decode().strip()
            if k16k17 != "":
                return arg, k16k17

        found_k = False
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(solve_k16k17, arg): arg for arg in args}
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    print(res)
                    if res is None: continue
                    arg, k16k17 = res
                    found_k = True
                    elapsed("k16k17 found")

                    break
                except Exception as exc:
                    print(f"Task {futures[future]} generated an exception: {exc}")

            executor.shutdown(wait=False)  # 他のプロセスを終了させる

        elapsed("end")
        if not found_k:
            print(":(")
            continue

        print(arg)
        parsed = json.loads(arg)
        k19 = parsed["k19"]
        k21 = parsed["k21"]
        k16, k17 = map(int, k16k17.split())
        print("k16k17:", k16k17)
        print("k19:", k19)
        print("k21:", k21)
        pt, ct = pairs[0]

        ALL_KEY_PATH = "exploration3/target/release/exploration3"
        INPUT3_PATH = "./input3.json"
        OUTPUT3_PATH = "./output3.json"

        with open(INPUT3_PATH, "w") as fp:
            json.dump(
                {
                    # これらがすでに求まっていることを期待
                    "k16": k16,
                    "k17": k17,
                    "k18": k18,
                    "k19": k19,
                    "k20": k20,
                    "k21": k21,
                    "pt": pt,
                    "ct": ct,  # encrypt(pt) == ct となるペアをひとつ用意 (どちらも int)
                },
                fp,
            )

        elapsed("start all key")
        p = subprocess.run([ALL_KEY_PATH, INPUT3_PATH, OUTPUT3_PATH])
        elapsed("done")

        try:
            with open(OUTPUT3_PATH, "r") as fp:
                output = json.load(fp)
        except FileNotFoundError:
            print("Fail...")
            continue
        print(output)
        key = hex(output["key"])[2:]

        sock.sendlineafter("Get balance", "3")
        sock.sendlineafter("key_guess:", key)
        
        sock.interactive()
        break
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(e)
