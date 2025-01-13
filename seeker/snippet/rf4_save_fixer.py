#date: 2025-01-13T17:01:23Z
#url: https://api.github.com/gists/754eb2f2f0f290e794f435801a464fa6
#owner: https://api.github.com/users/8f00ff

import argparse
import sys
import os
import binascii
from enum import Enum
from pathlib import Path
from typing import Optional

def create_parser():
  parser = argparse.ArgumentParser(
    description="Rune Factory 4 Save File CRC Tool",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  %(prog)s check rf4_s01.sav                             - Check save file integrity
  %(prog)s fix rf4_s01.sav                               - Repair save file CRC
  %(prog)s verify rf4_s01.sav                            - Detailed save file verification
  %(prog)s gender rf4_s01.sav -s rf4_sys.sav female - Change player gender
    """
  )
  
  parser.add_argument(
    'action',
    choices=['check', 'fix', 'verify', 'gender'],
    help="Action to perform on the save file"
  )
  
  parser.add_argument(
    'filename',
    help="Path to the RF4 save file"
  )
  
  parser.add_argument(
    'gender',
    nargs='?',
    choices=['male', 'female'],
    help="Gender selection (for gender action)"
  )
  
  parser.add_argument(
    '-q', '--quiet',
    action='store_true',
    help="Suppress output messages"
  )
  
  parser.add_argument(
    '-v', '--verbose',
    action='store_true',
    help="Display additional details"
  )
  
  parser.add_argument(
    '-s', '--system-save',
    help="Path to rf4_sys.sav file (required for gender change)"
  )
  
  return parser

class SaveType(Enum):
  # Save types with (size, type, padding, is_sys)
  PC_SYS = (9304, "SPECIALSYS", 0x00, True)
  PC_SAV = (140424, "PCSAV", 0xE0, False)
  SWITCH_SAV = (140416, "SWITCHSAV", 0xE0, False)
  THREE_DS_SAV = (140288, "3DSSAV", 0xE0, False)
  UNKNOWN = (0, "Unknown", 0, False)
  
  def __init__(self, size, save_type, padding, is_sys):
    self.size = size
    self.save_type = save_type
    self.padding = padding
    self.is_sys = is_sys
  
  def is_player_save(self) -> bool:
    return not self.is_sys

def get_save_type(filesize: int) -> SaveType:
  return next((t for t in SaveType if t.size == filesize), SaveType.UNKNOWN)

def verify_crc32(filename: str, silent: bool = False) -> tuple[bool, Optional[int], Optional[int]]:
  try:
    with open(filename, 'rb') as savefile:
      filesize = os.path.getsize(filename)
      save_type = get_save_type(filesize)
      
      if save_type == SaveType.UNKNOWN:
        if not silent:
          print(f"Unrecognized file size: {filesize} bytes")
        return False, None, None
      
      if save_type.is_sys:
        savefile.seek(4)
      else:
        savefile.seek(0)
      existing_crc = int.from_bytes(savefile.read(4), 'little')
      
      if not silent:
        print(f"Current CRC: {hex(existing_crc)}")
      
      if save_type.is_sys:
        savefile.seek(8)
        data = savefile.read()
      else:
        savefile.seek(4)
        data = savefile.read(filesize - 4 - save_type.padding)
      
      crc = binascii.crc32(data) & 0xFFFFFFFF
      
      if not silent:
        print(f"Calculated CRC: {hex(crc)}")
        if crc == existing_crc:
          print("CRC verification successful")
        else:
          print("CRC mismatch detected")
      
      return True, existing_crc, crc
  
  except Exception as e:
    if not silent:
      print(f"Error processing {filename}: {e}")
    return False, None, None

def fix_crc32(filename: str) -> bool:
  try:
    _, existing_crc, calculated_crc = verify_crc32(filename, silent=True)
    if calculated_crc is None:
      print("Unable to calculate CRC for {filename}")
      return False
    
    if existing_crc == calculated_crc:
      print(f"CRC of {filename} is already correct: {hex(existing_crc)}")
      return True
    
    with open(filename, 'rb+') as savefile:
      filesize = os.path.getsize(filename)
      save_type = get_save_type(filesize)
      
      if save_type.is_sys:
        savefile.seek(4)
      savefile.write(calculated_crc.to_bytes(4, 'little'))
      savefile.flush()
      print(f"CRC of {filename} is updated to {hex(calculated_crc)}")
    return True
  
  except Exception as e:
    print(f"Error updating CRC in {filename}: {e}")
    return False

def set_gender(filename: str, gender: str, sys_filename: str = None) -> bool:
  save_slot = int(Path(filename).stem.split('_s')[1])
  if save_slot < 1 or save_slot > 20:
    print(f"Invalid save slot number: {save_slot:%02d}. Must be 01-20)")
    return False
  
  target_gender_value = 0x01 if gender.lower() == 'female' else 0x00
  changes_needed = False
  
  try:
    with open(filename, 'rb+') as savefile:
      filesize = os.path.getsize(filename)
      save_type = get_save_type(filesize)
      
      if save_type != SaveType.PC_SAV:
        print("Gender change is only supported for PC save files")
        return False
      
      if not save_type.is_player_save():
        print("Gender can only be changed in player save files")
        return False
      
      savefile.seek(0x36)
      current_gender_value = int.from_bytes(savefile.read(1), 'little')
      new_gender_value = (current_gender_value & ~1) | target_gender_value
      if current_gender_value == new_gender_value:
        print(f"Player save already set to {gender}")
      else:
        savefile.seek(0x36)
        savefile.write(bytes([new_gender_value]))
        savefile.flush()
        print(f"Gender set to {gender} in player save file {filename}")
        if not fix_crc32(filename):
          return False
      
    if sys_filename:
      with open(sys_filename, 'rb+') as sysfile:
        if get_save_type(os.path.getsize(sys_filename)) != SaveType.PC_SYS:
          print("Invalid system save file")
          return False
        
        slot_offset = 0x500 + ((save_slot - 1) * 0xA4)
        sysfile.seek(slot_offset)
        
        current_gender_value = int.from_bytes(sysfile.read(1), 'little')
        new_gender_value = (current_gender_value & ~1) | target_gender_value
        
        if current_gender_value == new_gender_value:
          print(f"System save already set to {gender}")
        else:
          sysfile.seek(slot_offset)
          sysfile.write(bytes([new_gender_value]))
          sysfile.flush()
          print(f"Gender set to {gender} in system save file {filename} for slot {save_slot}")
          if not fix_crc32(sys_filename):
            return False
  
  except Exception as e:
    print(f"Error modifying gender in {filename}: {e}")
    return False
  
  return True

def main():
  parser = create_parser()
  args = parser.parse_args()
  
  if not os.path.exists(args.filename):
      print(f"Error: File not found - {args.filename}")
      return 1
  
  if args.action in ['check', 'verify']:
      success, existing, calculated = verify_crc32(args.filename, silent=args.quiet)
      if not success:
          return 1
      
      if args.verbose:
          print(f"\nFile type: {get_save_type(os.path.getsize(args.filename)).name}")
          print(f"File size: {os.path.getsize(args.filename)} bytes")
      
      return 0 if existing == calculated else 1
  
  elif args.action == 'fix':
      return 0 if fix_crc32(args.filename) else 1
  
  elif args.action == 'gender':
      return 0 if set_gender(args.filename, args.gender, args.system_save) else 1

  else:
      parser.print_help()
      return 1
  
if __name__ == "__main__":
  sys.exit(main())
