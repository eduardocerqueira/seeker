#date: 2024-12-18T17:03:26Z
#url: https://api.github.com/gists/138d3b34fb62197302bb90a5a9ae953f
#owner: https://api.github.com/users/wpcarro

lock = threading.Lock()
state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def persist_state():
    print("Persisting state...")
    tmp = "/tmp/buffer.json"
    dst = "/tmp/dump.json"
    with lock:
        content = json.dumps(state)
    with open(tmp, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f)
    os.rename(tmp, dst)

def handle_signal(signum, _frame):
    print(f"Received signal: {signum}")
    match signum:
        case signal.SIGINT:
            persist_state()
            # Restore the default behavior and re-signal
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt
        case signal.SIGTERM:
            persist_state()
            # Restore the default behavior and re-signal
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            raise SystemExit
        case signal.SIGHUP:
            persist_state()
            # Restore the default behavior and re-signal
            signal.signal(signal.SIGHUP, signal.SIG_DFL)
            raise SystemExit
        case x:
            logger.error(f"Unhandled signal: {x}")

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGHUP, handle_signal)

def checkpoint():
    sleep = 5
    minute = 60 / sleep
    for _ in range(int(60 * minute)):
        persist_state()
        print("Sleeping...")
        time.sleep(sleep)

# Start background thread that checkpoints state
t = threading.Thread(target=lambda: checkpoint(), daemon=True)
t.start()

sleep = 5
minute = 60 / sleep
for _ in range(int(60 * minute)):
    print("Awaiting signal...")
    if random.choice([True, False]):
        with lock:
            state.append(random.choice(range(100)))
    time.sleep(sleep)