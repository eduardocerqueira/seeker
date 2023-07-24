#date: 2023-07-24T16:38:36Z
#url: https://api.github.com/gists/8f77f7041e111184939c58f645379a0f
#owner: https://api.github.com/users/schrodyn

import time
from typing import List
import pefile
from capstone import *
from capstone.x86 import *
import re
import struct

# SAMPLE_PATH = 'bin/enc_string_test.bin32'
SAMPLE_PATH = 'bin/2cd2f077ca597ad0ef234a357ea71558d5e039da9df9958d0b8bd0efa92e74c9.bin32'
# SAMPLE_PATH = 'bin/cheat.bin'

STACK_SIZE = 0x3000
CHUNK_SIZE = 0x400

# 128 bit pack/unpack
def pack_128(val):
    a = val & 0xFFFFFFFFFFFFFFFF
    b = (val >> 64) & 0xFFFFFFFFFFFFFFFF
    return struct.pack('<QQ', a, b)
def unpack_128(val):
    a, b = struct.unpack('<QQ', val)
    return a | (b << 64)

class Env:
    def __init__(self):
        self.stack = bytearray(STACK_SIZE)
        self.reg = [0]*X86_REG_ENDING

    def clear(self):
        self.stack = bytearray(STACK_SIZE)
        self.reg = [0] * X86_REG_ENDING

    # save data to the stack as little endian at the given offset
    def save_stack(self, offset, data, size):
        # if offset is negative, wrap around
        if offset < 0:
            offset = STACK_SIZE + offset
        if offset + size > STACK_SIZE:
            offset = offset % STACK_SIZE

        if size == 1:
            self.stack[offset] = data
        elif size == 2:
            self.stack[offset:offset+2] = struct.pack('<H', data)
        elif size == 4:
            self.stack[offset:offset+4] = struct.pack('<I', data)
        elif size == 8:
            self.stack[offset:offset+8] = struct.pack('<Q', data)
        elif size == 16:
            self.stack[offset:offset+16] = pack_128(data)


    # load data from the stack as little endian at the given offset
    def load_stack(self, offset, size):
        if offset < 0:
            offset = STACK_SIZE + offset
        if offset + size > STACK_SIZE:
            offset = offset % STACK_SIZE

        if size == 1:
            return self.stack[offset]
        elif size == 2:
            return struct.unpack('<H', self.stack[offset:offset+2])[0]
        elif size == 4:
            return struct.unpack('<I', self.stack[offset:offset+4])[0]
        elif size == 8:
            return struct.unpack('<Q', self.stack[offset:offset+8])[0]
        elif size == 16:
            return unpack_128(self.stack[offset:offset+16])

def setup_capstone():
    md = Cs(CS_ARCH_X86, CS_MODE_32)
    md.detail = True
    md.skipdata = True
    md.syntax = CS_OPT_SYNTAX_INTEL
    return md

# find all pxor instructions using regex, then goes up chunk size and disassembles
def find_all_pxor(md: Cs, pe: pefile.PE):
    txt_section = pe.sections[0]
    txt_data = txt_section.get_data()
    image_base = pe.OPTIONAL_HEADER.ImageBase
    section_rva = txt_section.VirtualAddress

    pxor_egg = b'\x66\x0F\xEF'
    pxor_size = 6

    scan_end = txt_data.rfind(pxor_egg)
    txt_data = txt_data[:scan_end+pxor_size]

    # get a chunk of instructions starting from the given offset
    def get_chunk(start, size):
        instructions = []
        for inst in md.disasm(txt_data[start-size:start+pxor_size], image_base + section_rva + start + pxor_size - size):
            # we only care about pxor and mov instructions
            if inst.mnemonic == 'pxor' or inst.mnemonic == 'mov' or inst.mnemonic == 'movaps':
                instructions.append(inst)
        # skip if no instructions
        if len(instructions) == 0:
            return []
        # skip if first instruction is not pxor
        if instructions[-1].mnemonic != 'pxor':
            return []
        return instructions

    # get pxor chunks
    chunks = []
    for m in re.finditer(pxor_egg, txt_data, re.DOTALL):
        scan_end = m.start()
        chunks.append(get_chunk(scan_end, CHUNK_SIZE))
    return chunks

#simple xor function
def xor(data, key):
    out = []
    for i in range(len(data)):
        out.append(data[i] ^ key[i % len(key)])
    return bytes(out)

def emulate_chunk(chunk : List[CsInsn], env: Env):
    inst : CsInsn  # auto complete
    strings_out = []
    for inst in chunk:

        r, w = inst.regs_access()
        #print(f'{hex(inst.address)}: {inst.mnemonic} {inst.op_str}, r: {r}, w: {w}')
        if inst.mnemonic == 'mov':

            # if first op is stack pointer and second op is register
            if len(r) == 2 and (r[0] == X86_REG_ESP or r[0] == X86_REG_EBP):
                #print(f'0x{inst.address:x}: write {hex(env.reg[r[1]])} to stack at {hex(inst.disp)}')
                env.save_stack(inst.disp, env.reg[r[1]], inst.operands[1].size)

            # if first op is register and second op is stack pointer
            elif len(r) == 1 and (r[0] == X86_REG_ESP or r[0] == X86_REG_EBP) and len(w) == 1:
                #print(f'0x{inst.address:x}: read {hex(env.load_stack(inst.disp, inst.operands[0].size))} from stack at {hex(inst.disp)}')
                env.reg[w[0]] = env.load_stack(inst.disp, inst.operands[0].size)

            # if first op is stack pointer and second op is immediate
            elif len(r) == 1 and (r[0] == X86_REG_ESP or r[0] == X86_REG_EBP) and inst.operands[1].type == X86_OP_IMM:
                #print(f'0x{inst.address:x}: write {hex(inst.operands[1].imm)} to stack at {hex(inst.disp)}')
                env.save_stack(inst.disp, inst.operands[1].imm, inst.operands[1].size)

            # if first op is register and second op is immediate
            elif len(w) == 1 and inst.operands[1].type == X86_OP_IMM:
                #print(f'0x{inst.address:x}: write {hex(inst.operands[1].imm)} to {hex(inst.operands[0].reg)}')
                env.reg[w[0]] = inst.operands[1].imm

            # if first op is stack pointer and second op is register
            elif len(r) == 2 and (r[0] == X86_REG_ESP or r[0] == X86_REG_EBP):
                #print(f'0x{inst.address:x}: write {hex(env.reg[r[1]])} to stack at {hex(inst.disp)}')
                env.save_stack(inst.disp, env.reg[r[1]], inst.operands[1].size)

        elif inst.mnemonic == 'movaps':

            # if first op is stack pointer and second op is register
            if len(r) == 2 and (r[0] == X86_REG_ESP or r[0] == X86_REG_EBP):
                #print(f'0x{inst.address:x}: write {hex(env.reg[r[1]])} to stack at {hex(inst.disp)}')
                env.save_stack(inst.disp, env.reg[r[1]], inst.operands[1].size)

            # if first op is register and second op is stack pointer
            elif len(r) == 1 and (r[0] == X86_REG_ESP or r[0] == X86_REG_EBP) and len(w) == 1:
                #print(f'0x{inst.address:x}: read {hex(env.load_stack(inst.disp, inst.operands[0].size))} from stack at {hex(inst.disp)}')
                env.reg[w[0]] = env.load_stack(inst.disp, inst.operands[0].size)

        # looking for pxor with:
        # "pxor xmm0, xmmword ptr [esp+0x10]"
        # "pxor xmm0, xmmword ptr [ebp-0x10]"
        # e.g.
        elif inst.mnemonic == 'pxor':
            # grab the two operand values
            val1 = env.reg[r[0]]
            val2 = env.load_stack(inst.disp, inst.operands[1].size)

            #print(f'0x{inst.address:x}: xor {hex(val1)} with {hex(val2)} to get {hex(val1 ^ val2)}')

            if val1 == 0 or val2 == 0:
                continue
            # pack them into 128 bit values
            data = pack_128(val1)
            key = pack_128(val2)
            out = xor(data, key)

            #print(f'0x{inst.address:x}: xor {data} with {key} to get {out}')

            env.reg[w[0]] = unpack_128(out)
            strings_out.append((inst.address, out))

        else:
            raise Exception(f'Unknown instruction {inst.mnemonic}')
    # strings_out = b''.join(strings_out)
    # strings_out = strings_out.split(b'\x00')
    return strings_out

pe = pefile.PE(SAMPLE_PATH)
md = setup_capstone()

t = time.time()

chunks = find_all_pxor(md, pe)

print(f"found {len(chunks)} chunks")

env = Env()
strings = []

# loop through each chunk and "emulate" it
for chunk in chunks:
    if len(chunk) == 0:
        continue
    # extend the strings list with the strings found in this chunk
    strings.extend(emulate_chunk(chunk, env))
    env.clear()

print(f"Benchmark {time.time() - t}")

# hack to remove duplicates
# strings = list(dict.fromkeys(strings))

for a, s in strings:
    # tidy up some of the strings
    s = s.rstrip(b'\x00')
    s = s.decode('utf-8', 'ignore')
    if len(s) == 0:
        continue
    print(f'0x{a:08x}: {s}')