#date: 2026-03-03T17:29:01Z
#url: https://api.github.com/gists/843e4869a6dda0cc288d0361ad4e936e
#owner: https://api.github.com/users/harryaskham

#!/usr/bin/env python3
"""
Patch R02_3.00.06_FasterRawValuesMOD.bin into an optimized accel-only firmware.

This script takes the FasterRawValuesMOD firmware and applies three optimizations:
1. NOPs the PPG (photoplethysmography) driver initialization — frees CPU cycles
2. NOPs the Blood and HRS BLE notification sends — only accelerometer data is sent
3. Sets the timer config byte to 0x04 for maximum stable rate (~40Hz)

The result is a firmware that streams accelerometer data at ~40Hz without crashing,
compared to the stock EvenFasterRawValuesMOD which crashes after <1 minute at similar rates
due to BLE TX queue overflow from sending 3 notifications per timer tick.

Usage:
    python3 patch_accel_only.py R02_3.00.06_FasterRawValuesMOD.bin

Output: R02_3.00.06_EvenFasterRawValuesMOD-AccOnly.bin

The input file is R02_3.00.06_FasterRawValuesMOD.bin which can be found at:
    https://github.com/atc1441/ATC_RF03_Ring

Firmware format:
    0x00-0xFF: OTA header (256 bytes)
        0x00: Magic 0x12345678
        0x04: CRC32 of body (0x100 onwards)
        0x08: Body length
        0x0C: Body length (duplicate)
        0x10: FW version string
        0x30: HW version string
    0x100+: ARM Cortex-M4 (Thumb2) firmware body

Patches applied (all offsets are absolute file offsets):
    0x9B0A: Config byte (MOVS r7, #imm) — timer period control
            Stock FasterMod: 0x20 (32) → ~4Hz
            This patch:      0x04 (4)  → ~40Hz

    0x10B1E: PPG init function call (BL instruction, 4 bytes)
             Replaced with two ARM NOP instructions (00 BF 00 BF)

    0x09C0A: Blood packet CRC computation (BL instruction, 4 bytes) → NOP
    0x09C16: Blood packet BLE notification send (BL instruction, 4 bytes) → NOP
    0x09C54: HRS packet CRC computation (BL instruction, 4 bytes) → NOP
    0x09C60: HRS packet BLE notification send (BL instruction, 4 bytes) → NOP

Total: 6 patch sites, 25 bytes changed (1 config byte + 24 bytes of NOPs)
CRC32 at offset 0x04 is recalculated after patching.
"""

import struct
import sys
import zlib
import os

# ─── Patch definitions ───────────────────────────────────────────────

CONFIG_BYTE_OFFSET = 0x9B0A     # MOVS r7, #imm — timer period
CONFIG_BYTE_VALUE  = 0x04       # ~40Hz (was 0x20 = ~4Hz in FasterMod)

# ARM Thumb NOP = 0xBF00, two NOPs = 4 bytes replacing a BL instruction
NOP4 = bytes([0x00, 0xBF, 0x00, 0xBF])

PATCHES = [
    # (offset, replacement_bytes, description)
    (0x10B1E, NOP4, "NOP PPG driver init (BL → NOP×2)"),
    (0x09C0A, NOP4, "NOP Blood packet CRC compute (BL → NOP×2)"),
    (0x09C16, NOP4, "NOP Blood BLE notification send (BL → NOP×2)"),
    (0x09C54, NOP4, "NOP HRS packet CRC compute (BL → NOP×2)"),
    (0x09C60, NOP4, "NOP HRS BLE notification send (BL → NOP×2)"),
]

# ─── Main ────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = os.path.join(
        os.path.dirname(input_path),
        "R02_3.00.06_EvenFasterRawValuesMOD-AccOnly.bin"
    )

    # Read input
    with open(input_path, 'rb') as f:
        fw = bytearray(f.read())

    # Validate header
    magic = struct.unpack_from('<I', fw, 0)[0]
    if magic != 0x12345678:
        print(f"ERROR: Invalid magic 0x{magic:08X} (expected 0x12345678)")
        print("This doesn't look like a Colmi R02 firmware file.")
        sys.exit(1)

    stored_crc = struct.unpack_from('<I', fw, 4)[0]
    body_crc = zlib.crc32(bytes(fw[0x100:])) & 0xFFFFFFFF
    fw_ver = fw[0x10:0x30].rstrip(b'\x00').decode('ascii', errors='replace')
    hw_ver = fw[0x30:0x40].rstrip(b'\x00').decode('ascii', errors='replace')

    print(f"Input:      {input_path}")
    print(f"Size:       {len(fw)} bytes")
    print(f"FW version: {fw_ver}")
    print(f"HW version: {hw_ver}")
    print(f"CRC32:      0x{stored_crc:08X} ({'OK' if stored_crc == body_crc else 'MISMATCH'})")
    print()

    # Verify config byte location
    if fw[CONFIG_BYTE_OFFSET + 1] != 0x27:
        print(f"WARNING: Byte at 0x{CONFIG_BYTE_OFFSET+1:04X} is 0x{fw[CONFIG_BYTE_OFFSET+1]:02X}, "
              f"expected 0x27 (MOVS r7 opcode)")
        print("The firmware layout may have changed. Proceed with caution.")

    # Apply config byte patch
    old_config = fw[CONFIG_BYTE_OFFSET]
    fw[CONFIG_BYTE_OFFSET] = CONFIG_BYTE_VALUE
    print(f"Config byte 0x{CONFIG_BYTE_OFFSET:04X}: 0x{old_config:02X} → 0x{CONFIG_BYTE_VALUE:02X} "
          f"(MOVS r7, #0x{CONFIG_BYTE_VALUE:02X})")

    # Apply NOP patches
    for offset, patch_bytes, desc in PATCHES:
        old = fw[offset:offset + len(patch_bytes)]
        fw[offset:offset + len(patch_bytes)] = patch_bytes
        print(f"Patch 0x{offset:05X}: {old.hex()} → {patch_bytes.hex()} — {desc}")

    # Recalculate CRC32
    new_crc = zlib.crc32(bytes(fw[0x100:])) & 0xFFFFFFFF
    struct.pack_into('<I', fw, 4, new_crc)
    print(f"\nCRC32:      0x{new_crc:08X} (recalculated)")

    # Write output
    with open(output_path, 'wb') as f:
        f.write(fw)

    print(f"\nOutput:     {output_path}")
    print(f"            {len(fw)} bytes")
    print()
    print("Flash this firmware using the ATC RF03 Ring flasher:")
    print("  https://atc1441.github.io/ATC_RF03_Writer.html")
    print()
    print("Expected behavior:")
    print("  - Accelerometer data streams at ~40Hz (was ~4Hz)")
    print("  - PPG/SpO2 sensors disabled (saves CPU)")
    print("  - Only accelerometer BLE notifications sent (was 3 per tick)")
    print("  - Stable operation (no crash after <1 minute)")


if __name__ == "__main__":
    main()