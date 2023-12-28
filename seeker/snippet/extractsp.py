#date: 2023-12-28T17:04:13Z
#url: https://api.github.com/gists/52ac22f7a344b2f4822cfb601806e4e4
#owner: https://api.github.com/users/Prof9

import shutil
import struct
import subprocess
from pathlib import Path

path_7zip = Path("C:\\Program Files\\7-Zip\\7z.exe")
path_lon_in = Path("LoN_ch1\\DoJa_5_1_files\\LoN")
path_pon_in = Path("PoN_ch1to3\\DoJa_5_1_files\\PoN")
path_lon_out = Path("LoN_sp")
path_pon_out = Path("PoN_sp")


SP_HDR_SIZE = 0x40
PON_UNCOMPRESSED_FILES = [
    False,
    True,
    False,
    True,
    False,
    False,
    False,
    False,
    True,
    False,
    False,
    False,
    True,
    True,
    False,
    False,
    True,
    False,
    True,
    False,
    True,
    False,
    False,
    False,
    False,
    False,
    True,
    False,
    True,
    False,
    False,
    False,
    True,
    False,
    True,
    True,
    False,
    False,
    False,
    False,
]
LON_UNCOMPRESSED_FILES = [
    False,
    False,
    False,
    True,
    False,
    False,
    False,
    False,
    True,
    False,
    False,
    False,
    True,
    True,
    False,
    False,
    True,
    False,
    True,
    False,
    True,
    True,
    False,
    False,
    False,
    False,
    True,
    False,
    True,
    False,
    False,
    False,
    True,
    False,
    True,
    True,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
]


def dump_sp(path_in: Path, path_out: Path, lon: bool):
    path_out.mkdir(parents=True, exist_ok=True)

    dat_count = 100 if lon else 75
    dat_audio_flg_idx = 53
    sp_file_offset = 0xE50 if lon else 0xDD8
    sys_dat_offset = 0x64000
    dat_file_count = 42 if lon else 37
    uncompressed_files = LON_UNCOMPRESSED_FILES if lon else PON_UNCOMPRESSED_FILES
    img0_file_idx = 1
    img0_file_count = 56 if lon else 42
    img1_file_idx = 2
    img1_file_count = 14 if lon else 16
    img2_file_idx = 0
    img2_file_count = 70 if lon else 56
    img3_file_idx = 3
    img3_file_count = 24 if lon else 22
    mld_file_idx = 5
    mld_file_count = 28 if lon else 16

    with open(path_in, "rb") as sp:
        # Read `dat`
        dat = []
        sp.seek(SP_HDR_SIZE + 0x0)
        for i in range(dat_count):
            dat.append(struct.unpack_from(">I", sp.read(4))[0])
        print(f"dat = {dat}")
        print(f"audio_flg = {dat[dat_audio_flg_idx]}")

        # Read `sys_dat`
        sys_dat = []
        sp.seek(SP_HDR_SIZE + sys_dat_offset)
        for i in range(255):
            sys_dat.append(struct.unpack_from(">I", sp.read(4))[0])
        print(f"sys_dat = {sys_dat}")

        # Read `mld` (melody)
        sp.seek(SP_HDR_SIZE + sys_dat[0])
        mld = sp.read(sys_dat[1])
        with open(path_out / "sp_mld.zip", "wb") as mld_f:
            mld_f.write(mld)
        subprocess.run(
            [
                path_7zip,
                "x",
                path_out / f"sp_mld.zip",
                f"-o{path_out}\\sp_mld",
                "-aoa",
            ]
        ).check_returncode()
        shutil.move(path_out / f"sp_mld" / "data.dat", path_out / f"sp_mld.bin")
        shutil.rmtree(path_out / f"sp_mld")
        (path_out / f"sp_mld.zip").unlink()
        
        # Dump mld
        with open(path_out / f"sp_mld.bin", "rb") as mld_f:
            # Parse header
            file_count, = struct.unpack_from(">I", mld_f.read(4))
            sizes = []
            for i in range(file_count):
                size, = struct.unpack_from(">I", mld_f.read(4))
                sizes.append(size)
            
            # Extract individual melodies
            (path_out / "sp_mld").mkdir(parents=True, exist_ok=True)
            for i in range(file_count):
                mld_bytes = mld_f.read(sizes[i])
                (path_out / "sp_mld" / f"mld_{i}.mld").write_bytes(mld_bytes)

        # Compute file addresses
        file_addrs = []
        file_addrs.append(sp_file_offset)
        for i in range(1, dat_file_count):
            file_addrs.append(file_addrs[-1] + dat[i - 1])

        # Dump jar files
        for i in range(dat_file_count):
            if dat[i] == 0:
                continue
            sp.seek(SP_HDR_SIZE + file_addrs[i])
            f = sp.read(dat[i])
            if uncompressed_files[i]:
                (path_out / f"sp_jar{i}.bin").write_bytes(f)
            else:
                (path_out / f"sp_jar{i}.zip").write_bytes(f)
                (path_out / f"sp_jar{i}").mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        path_7zip,
                        "x",
                        path_out / f"sp_jar{i}.zip",
                        f"-o{path_out}\\sp_jar{i}",
                        "-aoa",
                    ]
                ).check_returncode()
                shutil.move(
                    path_out / f"sp_jar{i}" / "data.dat", path_out / f"sp_jar{i}.bin"
                )
                shutil.rmtree(path_out / f"sp_jar{i}")
                (path_out / f"sp_jar{i}.zip").unlink()

        # Dump menu sprites
        with open(path_out / f"sp_jar{img0_file_idx}.bin", "rb") as f:
            sizes = []
            for i in range(img0_file_count):
                (size,) = struct.unpack_from(">I", f.read(4))
                sizes.append(size)
            (path_out / f"sp_jar{img0_file_idx}").mkdir(parents=True, exist_ok=True)
            for i in range(img0_file_count):
                img = f.read(sizes[i])
                (path_out / f"sp_jar{img0_file_idx}" / f"img_{i}.gif").write_bytes(img)

        # Dump battle sprites
        with open(path_out / f"sp_jar{img1_file_idx}.bin", "rb") as f:
            sizes = []
            for i in range(img1_file_count):
                (size,) = struct.unpack_from(">I", f.read(4))
                sizes.append(size)
            (path_out / f"sp_jar{img1_file_idx}").mkdir(parents=True, exist_ok=True)
            for i in range(img1_file_count):
                img = f.read(sizes[i])
                (path_out / f"sp_jar{img1_file_idx}" / f"img_{i}.gif").write_bytes(img)

        # Dump sprite metadata
        with open(path_out / f"sp_jar{img2_file_idx}.bin", "rb") as f:
            sizes = []
            for i in range(img2_file_count):
                (size,) = struct.unpack_from(">I", f.read(4))
                sizes.append(size)
            (path_out / f"sp_jar{img2_file_idx}").mkdir(parents=True, exist_ok=True)
            for i in range(img2_file_count):
                img = f.read(sizes[i])
                (path_out / f"sp_jar{img2_file_idx}" / f"img_{i}.bin").write_bytes(img)

        # Dump real world backgrounds
        with open(path_out / f"sp_jar{img3_file_idx}.bin", "rb") as f:
            sizes = []
            for i in range(img3_file_count):
                (size,) = struct.unpack_from(">I", f.read(4))
                sizes.append(size)
            (path_out / f"sp_jar{img3_file_idx}").mkdir(parents=True, exist_ok=True)
            for i in range(img3_file_count):
                img = f.read(sizes[i])
                (path_out / f"sp_jar{img3_file_idx}" / f"img_{i}.gif").write_bytes(img)

        # Dump sound
        with open(path_out / f"sp_jar{mld_file_idx}.bin", "rb") as f:
            sizes = []
            for i in range(mld_file_count):
                (size,) = struct.unpack_from(">I", f.read(4))
                sizes.append(size)
            (path_out / f"sp_jar{mld_file_idx}").mkdir(parents=True, exist_ok=True)
            for i in range(mld_file_count):
                img = f.read(sizes[i])
                (path_out / f"sp_jar{mld_file_idx}" / f"mld_{i}.mld").write_bytes(img)


dump_sp(path_pon_in / "sp" / "PoN.sp", path_pon_out, False)
dump_sp(path_lon_in / "sp" / "LoN.sp", path_lon_out, True)
