#date: 2024-11-19T16:43:25Z
#url: https://api.github.com/gists/21c95c1a875e219ebb1c956569c2dd8f
#owner: https://api.github.com/users/Ailyth99

###################################################################
# ARC file unpacker for PS2 Kagero 2 / Trapt game's arc files
###################################################################
import os
import struct
from pathlib import Path

def is_arc_file(file_path):
    """Check if file is ARC format"""
    try:
        with open(file_path, 'rb') as f:
            magic = f.read(4)
            return magic == b'RPS\x00'
    except:
        return False

def unpack_arc(input_file, output_dir):
    """Unpack single ARC file"""
    try:
        # Create subfolder named after file
        file_name = Path(input_file).stem
        sub_output_dir = os.path.join(output_dir, file_name)
        os.makedirs(sub_output_dir, exist_ok=True)

        with open(input_file, 'rb') as f:
            # Check file header
            magic = f.read(4)
            if magic != b'ARC\x00':
                raise ValueError(f"Not a valid ARC file: {input_file}")

            # Read header info
            files_count = struct.unpack('<I', f.read(4))[0]
            fstart_off = struct.unpack('<I', f.read(4))[0]
            arc_size = struct.unpack('<I', f.read(4))[0]
            arc_size2 = struct.unpack('<I', f.read(4))[0]
            zero = struct.unpack('<I', f.read(4))[0]

            # Process each file
            for i in range(files_count):
                # Read file info
                size = struct.unpack('<I', f.read(4))[0]
                size2 = struct.unpack('<I', f.read(4))[0]
                name_off = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<I', f.read(4))[0]
                
                if i < files_count - 1:
                    zero = struct.unpack('<I', f.read(4))[0]

                current_pos = f.tell()

                # Read filename
                f.seek(name_off)
                name = b''
                while True:
                    char = f.read(1)
                    if char == b'\x00':
                        break
                    name += char
                name = name.decode('utf-8')

                # Read file data
                f.seek(offset + fstart_off)
                data = f.read(size)

                # Save file
                output_path = os.path.join(sub_output_dir, name)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'wb') as out_file:
                    out_file.write(data)

                f.seek(current_pos)

            print(f"Successfully unpacked {files_count} files from {input_file} to {sub_output_dir}")
            return True
    except Exception as e:
        print(f"Error unpacking {input_file}: {str(e)}")
        return False

def batch_unpack(input_dir, output_dir):
    """Batch unpack all RPS files in folder"""
    os.makedirs(output_dir, exist_ok=True)
    
    total_files = 0
    successful_files = 0
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            total_files += 1
            
            if is_arc_file(file_path):
                if unpack_arc(file_path, output_dir):
                    successful_files += 1
            else:
                print(f"Skipping non-ARC file: {file_path}")
    
    print(f"\nBatch processing complete!")
    print(f"Total files: {total_files}")
    print(f"Successfully unpacked: {successful_files}") 
    print(f"Skipped files: {total_files - successful_files}")

# Usage example
if __name__ == "__main__":
    input_directory = r""  # Input folder path
    output_directory = ""  # Output folder path
    batch_unpack(input_directory, output_directory)