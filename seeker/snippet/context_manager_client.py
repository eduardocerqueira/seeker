#date: 2023-12-18T17:01:12Z
#url: https://api.github.com/gists/63ad551073593a5754d4e1eb417d1f72
#owner: https://api.github.com/users/Adoni5

from contextlib import contextmanager, redirect_stdout
from pyguppy_client_lib import helper_functions
from pyguppy_client_lib.pyclient import PyGuppyClient
BIN_PATH = "/usr/bin/"
  
  
@contextmanager
def start_guppy_server_and_client(bin_path, config, port, server_args):
    server_args.extend(
        ["--config", config, "--port", port, "--log_path", str((Path(".") / "junk"))]
    )
    # This function has it's own prints that may want to be suppressed
    with redirect_stdout(StringIO()) as fh:
        server, port = helper_functions.run_server(server_args, bin_path=bin_path)

    if port == "ERROR":
        raise RuntimeError("Server couldn't be started")

    if port.startswith("ipc"):
        address = f"{port}"
    client = PyGuppyClient(address=address, config=config)

    try:
        with client:
            yield client
    finally:
        server.terminate()

# /Example use
with start_guppy_server_and_client(
    BIN_PATH,
    "dna_r10.4.1_e8.2_400bps_hac.cfg",
    "ipc:///tmp/.guppy/5555",
    ["--device", "cuda:all"],
) as client:
#   Reads the 
  for channel, read in reads:
    # Attach the "RF-" prefix
    read_id = f"RF-{read.id}"
    t0 = time.time()
    cache[read_id] = (channel, read.number, t0)
    success = self.caller.pass_read(
        package_read(
            read_id=read_id,
            raw_data=np.frombuffer(read.raw_data, signal_dtype),
            daq_offset=daq_values[channel].offset,
            daq_scaling=daq_values[channel].scaling,
        )
    )
    if not success:
        logging.warning(f"Could not send read {read_id!r} to Guppy")
        # FIXME: This is resolved in later versions of guppy.
        skipped[read_id] = cache.pop(read_id)
        continue
    else:
        reads_sent += 1

    sleep_time = self.caller.throttle - t0
    if sleep_time > 0:
        time.sleep(sleep_time)

  while reads_received < reads_sent:
    results = self.caller.get_completed_reads()
    # TODO: incorporate time_received into logging?
    # time_received = time.time()

    if not results:
        time.sleep(self.caller.throttle)
        continue

    for res_batch in results:
        for res in res_batch:
            read_id = res["metadata"]["read_id"]
            try:
                channel, read_number, time_sent = cache.pop(read_id)
            except KeyError:
                # FIXME: This is resolved in later versions of guppy.
                channel, read_number, time_sent = skipped.pop(read_id)
                reads_sent += 1
            res["metadata"]["read_id"] = read_id[3:]
            self.logger.debug(
                "@%s ch=%s\n%s\n+\n%s",
                res["metadata"]["read_id"],
                channel,
                res["datasets"]["sequence"],
                res["datasets"]["qstring"],
            )
            barcode = res["metadata"].get("barcode_arrangement", None)
            # TODO: Add Filter here
            yield Result(
                channel=channel,
                read_number=read_number,
                read_id=res["metadata"]["read_id"],
                seq=res["datasets"]["sequence"],
                barcode=barcode if barcode else None,
                basecall_data=res,
            )
            reads_received += 1