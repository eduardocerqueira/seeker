#date: 2024-05-08T17:01:14Z
#url: https://api.github.com/gists/f2c76769e2a06e7317ba04753199ba8a
#owner: https://api.github.com/users/merrymercy

# mpirun -np 2 python p2p-nonblocking.py

import cupy as cp
import cupy.cuda.nccl as nccl
from mpi4py import MPI
import time
import os

nbytes = 1024*1024*32
data_type = cp.float32
buffsize = nbytes
os.environ["NCCL_BUFFSIZE"] = str(buffsize)
os.environ["NCCL_P2P_NVL_CHUNKSIZE"] = str(buffsize)
os.environ["NCCL_P2P_NET_CHUNKSIZE"] = str(buffsize)
os.environ["NCCL_MAX_NCHANNELS"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"

def run_benchmark(mpi_comm, nccl_comm):
    if data_type == cp.float32:
        nccl_dtype = nccl.NCCL_FLOAT32
        nbytes_per_elem = 4
    nelem = nbytes // nbytes_per_elem

    memory = cp.zeros(nelem, dtype=data_type)


    stream = cp.cuda.Stream(non_blocking=True)

    # warmup to just make the connections
    nccl.groupStart()
    if mpi_comm.rank == 0:
        nccl_comm.send(
            memory.data.ptr, nelem, nccl_dtype, 1, stream.ptr
        )
    elif mpi_comm.rank == 1:
        nccl_comm.recv(
            memory.data.ptr, nelem, nccl_dtype, 0, stream.ptr
        )
    nccl.groupEnd()

    a = cp.ones((128,))
    b = cp.ones((128,))

    cp.cuda.runtime.deviceSynchronize()
    mpi_comm.barrier()
    st = time.time()

    if mpi_comm.rank == 0:
        pass
        #nccl_comm.send(
        #    memory.data.ptr, nelem, nccl_dtype, 1, stream.ptr
        #)
    elif mpi_comm.rank == 1:
        print("recv begin")
        nccl_comm.recv(
            memory.data.ptr, nelem, nccl_dtype, 0, stream.ptr
        )
        print("recv end")

        print("compute begin")
        with cp.cuda.Stream():
            c = a + b
        print("compute end")

    cp.cuda.runtime.deviceSynchronize()
    en = time.time()
    print(f"{mpi_comm.rank} took {en-st} seconds")

def create_nccl_comm(mpi_comm):
    root = 0
    if mpi_comm.rank == root:
        uid = nccl.get_unique_id()
    else:
        uid = None
    uid = mpi_comm.bcast(uid, root=root)

    cp.cuda.runtime.deviceSynchronize()
    tic = time.time()

    comm = nccl.NcclCommunicator(mpi_comm.size, uid, mpi_comm.rank)

    cp.cuda.runtime.deviceSynchronize()
    print(f"communicator cost: {time.time() - tic:.2f}s")

    return comm


if __name__ == "__main__":
  world_comm = MPI.COMM_WORLD
  world_rank = world_comm.rank
  world_size = world_comm.size
  nccl_comm = None
  assert world_size == 2
  try:
    cp.cuda.Device(world_rank).use()

    nccl_comm = create_nccl_comm(world_comm)

    run_benchmark(world_comm, nccl_comm)

    nccl_comm = None
    MPI.Finalize()

    world_comm = None

  except Exception as e:
    print(f"An error occurred: {e}")
    if nccl_comm:
        nccl_comm.abort()
    world_comm.Abort()