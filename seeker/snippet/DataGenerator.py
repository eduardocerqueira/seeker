#date: 2023-02-13T16:52:50Z
#url: https://api.github.com/gists/426f0477a7c54e91362533474cf0f32a
#owner: https://api.github.com/users/MCardus

import os
import sys

class DataGenerator:
    def copy_files(self, src_dir: str, dest_dir: str) -> None:
        try:
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            for filename in os.listdir(src_dir):
                src_file = os.path.join(src_dir, filename)
                dest_file = os.path.join(dest_dir, filename)

                if os.path.isdir(src_file):
                    self.copy_files(src_file, dest_file)
                else:
                     self._copy_first_line(src_file,dest_file )  
        except Exception as e: 
            print(f"Something went wrong with {src_file} in line {e.__traceback__.tb_lineno}\n{e}")
 

    def _read_first_line(self, path: str) -> str:
        with open(path, 'r') as f:
            return f.readline()

    def _copy_first_line(self, src_file: str, dest_file: str) -> None:
        first_line = self._read_first_line(src_file)
        with open(dest_file, 'w') as dest_f:
            dest_f.write(first_line)

if __name__ == '__main__': 
    src_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    print(f"Moving data from {src_dir} to {dest_dir}")
    DataGenerator().copy_files(src_dir, dest_dir)
