#date: 2023-09-25T16:52:58Z
#url: https://api.github.com/gists/5c1485e689a1866988758bf62b03a43a
#owner: https://api.github.com/users/rxwx

from ctypes import wintypes
import argparse
import ctypes
import yara
import hexdump

"""
.text:0000000180010840                         ; char __fastcall BeaconDataStoreUnprotectItem(unsigned __int64)
.text:0000000180010840                         BeaconDataStoreUnprotectItem proc near  ; CODE XREF: sub_1800100F8+9E↑p
.text:0000000180010840                                                                 ; sub_1800102E8+AD↑p
.text:0000000180010840                                                                 ; DATA XREF: ...
.text:0000000180010840 48 3B 0D A9 8C 03 00                    cmp     rcx, cs:beacon_data_store_size
.text:0000000180010847 73 55                                   jnb     short locret_18001089E
.text:0000000180010849 4C 8B 05 98 8C 03 00                    mov     r8, cs:beacon_datastore
.text:0000000180010850 48 8D 14 89                             lea     rdx, [rcx+rcx*4]
.text:0000000180010854 45 33 DB                                xor     r11d, r11d
.text:0000000180010857 45 39 1C D0                             cmp     [r8+rdx*8], r11d
.text:000000018001085B 74 41                                   jz      short locret_18001089E
.text:000000018001085D 45 39 5C D0 10                          cmp     [r8+rdx*8+10h], r11d
.text:0000000180010862 74 3A                                   jz      short locret_18001089E
.text:0000000180010864 4D 39 5C D0 18                          cmp     [r8+rdx*8+18h], r11
.text:0000000180010869 74 33                                   jz      short locret_18001089E
.text:000000018001086B 45 8B CB                                mov     r9d, r11d
.text:000000018001086E 4D 39 5C D0 20                          cmp     [r8+rdx*8+20h], r11
.text:0000000180010873 76 24                                   jbe     short loc_180010899
.text:0000000180010875
.text:0000000180010875                         loc_180010875:                          ; CODE XREF: BeaconDataStoreUnprotectItem+57↓j
.text:0000000180010875 49 8B 4C D0 18                          mov     rcx, [r8+rdx*8+18h]
.text:000000018001087A 49 8B C1                                mov     rax, r9
.text:000000018001087D 4C 8D 15 74 8C 03 00                    lea     r10, byte_1800494F8
.text:0000000180010884 83 E0 03                                and     eax, 3
.text:0000000180010887 42 8A 04 10                             mov     al, [rax+r10]
.text:000000018001088B 42 30 04 09                             xor     [rcx+r9], al
.text:000000018001088F 49 FF C1                                inc     r9
.text:0000000180010892 4D 3B 4C D0 20                          cmp     r9, [r8+rdx*8+20h]
.text:0000000180010897 72 DC                                   jb      short loc_180010875
.text:0000000180010899
.text:0000000180010899                         loc_180010899:                          ; CODE XREF: BeaconDataStoreUnprotectItem+33↑j
.text:0000000180010899 45 89 5C D0 10                          mov     [r8+rdx*8+10h], r11d
.text:000000018001089E
.text:000000018001089E                         locret_18001089E:                       ; CODE XREF: BeaconDataStoreUnprotectItem+7↑j
.text:000000018001089E                                                                 ; BeaconDataStoreUnprotectItem+1B↑j ...
.text:000000018001089E C3                                      retn
.text:000000018001089E                         BeaconDataStoreUnprotectItem endp
"""

# https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-memory_basic_information
class MEMORY_BASIC_INFORMATION(ctypes.Structure):

    _fields_ = [("BaseAddress", wintypes.LPVOID),
                ("AllocationBase", wintypes.LPVOID),
                ("AllocationProtect", wintypes.DWORD),
                ("PartitionId", wintypes.WORD),
                ("RegionSize", ctypes.c_size_t),
                ("State", wintypes.DWORD),
                ("Protect", wintypes.DWORD),
                ("Type", wintypes.DWORD)]

    # Allow this structure to print itself
    def __repr__(self):
        return f'MEMORY_BASIC_INFORMATION(BaseAddress={self.BaseAddress if self.BaseAddress is not None else 0:#x}, ' \
                                        f'AllocationBase={self.AllocationBase if self.AllocationBase is not None else 0:#x}, ' \
                                        f'AllocationProtect={self.AllocationProtect:#x}, ' \
                                        f'PartitionId={self.PartitionId:#x}, ' \
                                        f'RegionSize={self.RegionSize:#x}, ' \
                                        f'State={self.State:#x}, ' \
                                        f'Protect={self.Protect:#x}, ' \
                                        f'Type={self.Type:#x})'

PMEMORY_BASIC_INFORMATION = ctypes.POINTER(MEMORY_BASIC_INFORMATION)

ReadProcessMemory = ctypes.WinDLL('kernel32', use_last_error=True).ReadProcessMemory
ReadProcessMemory.argtypes = [wintypes.HANDLE, wintypes.LPCVOID, wintypes.LPVOID, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
ReadProcessMemory.restype = wintypes.BOOL

VirtualQueryEx = ctypes.WinDLL('kernel32', use_last_error=True).VirtualQueryEx
VirtualQueryEx.argtypes = [wintypes.HANDLE, wintypes.LPCVOID, PMEMORY_BASIC_INFORMATION, ctypes.c_size_t]
VirtualQueryEx.restype = ctypes.c_size_t

PAGE_READWRITE = 0x04
PAGE_EXECUTE_READ = 0x20

MEM_COMMIT = 0x00001000
MEM_IMAGE = 0x1000000
MEM_MAPPED = 0x40000
MEM_PRIVATE = 0x20000

class DATA_STORE_OBJECT(ctypes.Structure):

    _fields_ = [("type", wintypes.INT),
                ("hash", ctypes.c_ulonglong),
                ("masked", wintypes.BOOL),
                ("buffer", wintypes.LPVOID),
                ("length", ctypes.c_size_t)]

    def __repr__(self):
        return f'DATA_STORE_OBJECT(type={self.type if self.type is not None else 0:#x}, ' \
                                        f'hash={self.hash if self.hash is not None else 0:#x}, ' \
                                        f'masked={self.masked:#x}, ' \
                                        f'buffer={self.buffer if self.buffer is not None else 0:#x}, ' \
                                        f'length={self.length:#x})'

PDATA_STORE_OBJECT = ctypes.POINTER(DATA_STORE_OBJECT)

YARA_BeaconDataStoreUnprotectItem = """
rule CobaltStrike_BeaconDataStoreUnprotectItem {
    strings:
        $a_x64 = { 48 3B 0D A9 8C 03 00 73 55 4C 8B 05 98 8C 03 00 48 8D 14 89 45 33 DB 45 39 1C D0 74 41 }
    condition:
        any of them
}
"""

READ_BUF_SIZE = 1024
MATCH_BYTES = b'\x48\x3B\x0D\xA9\x8C\x03\x00\x73\x55\x4C\x8B\x05\x98\x8C\x03\x00\x48\x8D\x14\x89\x45\x33\xDB\x45\x39\x1C\xD0\x74\x41'

def unmask_data(buf, key):
    outbuf = bytearray(len(buf))
    for i in range(0, len(buf)):
        result = key[i & 3]
        outbuf[i] = buf[i] ^ result
    return bytes(outbuf)

def read_pointer(pid, addr):
    hProc = ctypes.windll.kernel32.OpenProcess(0x1fffff, False, pid)
    if not hProc:
        print (f"Unable to open process: {pid}")
        return

    ptr = ctypes.c_void_p(0)
    bytes_read = ctypes.c_size_t(0)
    if not ReadProcessMemory(hProc, addr, ctypes.byref(ptr), ctypes.sizeof(ctypes.c_void_p), bytes_read):
        print (f"Unable to read pointer at: {addr}")
        return 0
    
    ctypes.windll.kernel32.CloseHandle(hProc)
    return ptr.value

def dump_datastore(pid, dsSizeAddr, dsAddr, keyAddr):
    hProc = ctypes.windll.kernel32.OpenProcess(0x1fffff, False, pid)
    if not hProc:
        print (f"Unable to open process: {pid}")
        return
    
    # Read the key
    keybuf = ctypes.c_buffer(8)
    bytes_read = ctypes.c_size_t(0)
    if not ReadProcessMemory(hProc, keyAddr, keybuf, 8, ctypes.byref(bytes_read)):
        print ("Unable to read key")
        ctypes.windll.kernel32.CloseHandle(hProc)
        return
    
    print (f"Read key: {keybuf.raw.hex()}")
    
    # Read the datastore entries
    dsSize = ctypes.c_long(0)
    if not ReadProcessMemory(hProc, dsSizeAddr, ctypes.byref(dsSize), ctypes.sizeof(ctypes.c_int), ctypes.byref(bytes_read)):
        print ("Unable to get size of Beacon data store")
        return
    
    print (f"Datastore size: {dsSize.value}")
    buffer = ctypes.c_buffer(ctypes.sizeof(DATA_STORE_OBJECT))
    for x in range(0, dsSize.value):
        addr = read_pointer(pid, dsAddr)
        if not addr:
            return
        
        addr = addr + (x * ctypes.sizeof(DATA_STORE_OBJECT))
        if ReadProcessMemory(hProc, addr, buffer, ctypes.sizeof(DATA_STORE_OBJECT), ctypes.byref(bytes_read)):
            entry = DATA_STORE_OBJECT.from_buffer(buffer)
            print(entry)
            if entry.type and entry.masked:
                # read the masked buffer
                bytes_read.value = 0
                masked = ctypes.c_buffer(entry.length)
                if not ReadProcessMemory(hProc, entry.buffer, masked, entry.length, ctypes.byref(bytes_read)):
                    print (f"Unable to read {entry.length} bytes from datastore slot {x} at {entry.buffer}")
                    continue

                # Decrypt the data stored at entry->buffer
                assert len(masked) == entry.length
                unmasked = unmask_data(masked.raw, keybuf.raw)

                # print/write output
                print (f"Umasked data sample:")
                hexdump.hexdump(unmasked[:100])
                fname = f'beacon_datastore_entry_{x}_{addr}.bin'
                with open(fname, 'wb') as f:
                    f.write(unmasked)
                print (f"Written to: {fname}")

    ctypes.windll.kernel32.CloseHandle(hProc)

def get_datastore_addr(pid):
    matchAddr = 0
    hProc = ctypes.windll.kernel32.OpenProcess(0x1fffff, False, pid)
    if hProc:
        print (f"Got handle: {hProc}")
        mbi = MEMORY_BASIC_INFORMATION()
        address = 0
        while not matchAddr and VirtualQueryEx(hProc, address, ctypes.byref(mbi), ctypes.sizeof(mbi)):
            if mbi.Protect == PAGE_EXECUTE_READ and mbi.AllocationProtect == PAGE_READWRITE \
                    and mbi.State == MEM_COMMIT and mbi.Type == MEM_PRIVATE:
                print (f"Searching region: {mbi.BaseAddress:#x}")
                curAddr = mbi.BaseAddress
                buffer = ctypes.c_buffer(READ_BUF_SIZE)
                bytes_read = ctypes.c_size_t(0)
                while curAddr < mbi.BaseAddress + mbi.RegionSize:
                    if ReadProcessMemory(hProc, curAddr, buffer, READ_BUF_SIZE, ctypes.byref(bytes_read)):
                        if MATCH_BYTES in buffer.raw:
                            matchAddr = curAddr + buffer.raw.index(MATCH_BYTES)
                            print ("Found BeaconDataStoreUnprotectItem at: {0:#x}".format(matchAddr))
                            break
                    curAddr += READ_BUF_SIZE
            address += mbi.RegionSize
        ctypes.windll.kernel32.CloseHandle(hProc)
    return matchAddr

def dump_beacon(pid):
    rules = yara.compile(source=YARA_BeaconDataStoreUnprotectItem)
    matches = rules.match(pid=pid)
    if (matches):
        print ('Found CobaltStrike 4.9+ beacon with Datastore')
        funcAddr = get_datastore_addr(pid)
        if funcAddr <= 0:
            print ("Unable to locate BeaconDataStoreUnprotectItem function")
            return

        # TODO: resolve offsets dynamically from opcodes
        dsSizeAddr = funcAddr + 0x38cb0     # cmp    rcx,QWORD PTR [rip+0x38ca9]
        dsAddr = funcAddr + 0x38ca8         # mov    r8,QWORD PTR [rip+0x38c98]
        keyAddr = funcAddr + 0x38cb8        # lea    r10,[rip+0x38c74]

        print (f"Datastore size address: {dsSizeAddr:#x}")
        print (f"Datastore address: {dsAddr:#x}")
        print (f"Key address: {keyAddr:#x}")
        dump_datastore(pid, dsSizeAddr, dsAddr, keyAddr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dump encrypted data from a CobaltStrike beacon datastore')
    parser.add_argument("pid", help="PID of CobaltStrike beacon", type=int)
    args = parser.parse_args()

    dump_beacon(args.pid)
