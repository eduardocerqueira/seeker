#date: 2023-02-23T16:59:59Z
#url: https://api.github.com/gists/b0216911fe295dd24b566940b12cf094
#owner: https://api.github.com/users/BluBb-mADe

import os, sys, shutil, ctypes, time, re, operator

# Usage:
# This script always makes sure it makes backups of everything it changes so you can always revert by renaming the backup files to their original names.
# If you only have one version of unity installed it will just search for the first unity version it can find there and try to patch that if you run it.
# You can adjust the default_path variable inside this script if necessary.
# Alternatively you can just drag and drop the Unity.exe onto this script if you have properly associated the python script file extension with a working interpreter.
# Or you can pass the path to the Unity.exe as the only argument to the script in a shell by hand.
#
# I have only tested this script on windows and I haven't checked if even the signatures could work on other platforms. But probably not.

patches = (
    # ("2021.2.2",  '4c 8d 05 1e 76 f7 00 48 8d 55 84 48 8b 4c 24 60 e8 c0 b0 43 fe 90', '0F B6 C3', 'B0 01 90'),
    # ("2021.2.1",  '4c 8d 05 fe 68 f7 00 48 8d 55 84 48 8b 4c 24 60 e8 e0 d6 43 fe 90', '0F B6 C3', 'B0 01 90'),
    # ("2021.2.0",  '4c 8d 05 4e 61 f7 00 48 8d 55 84 48 8b 4c 24 60 e8 f0 72 44 fe 90', '0F B6 C3', 'B0 01 90'),
    # ("2021.1.28", '48 8d 4d 18 e8 99 d0 7d fe 90 48 8d 4c 24 60 e8 8e d0 7d fe 90', '0F B6 C3', 'B0 01 90'),
    # ("2021.1.5",  '48 8d 4d 18 e8 e9 ae 7c fe 90 48 8d 4c 24 60 e8 de ae 7c fe 90', '0F B6 C3', 'B0 01 90'),
    # ("2021.1.0",  '48 8d 4d 18 e8 19 54 7d fe 90 48 8d 4c 24 60 e8 0e 54 7d fe 90', '0F B6 C3', 'B0 01 90'),
    # ("2020.3.22", '48 8d 4d 28 e8 b0 ef b5 fc 90 48 8d 4d b8 e8 a6 ef b5 fc 90', '0F B6 C3', 'B0 01 90'),
    # ("2020.3.0",  '48 8d 4d 28 e8 c0 fa c2 fc 90 48 8d 4d b8 e8 b6 fa c2 fc 90', '0F B6 C3', 'B0 01 90'),
    # ("2020.2.7",  '48 8d 4d 28 e8 50 58 c3 fc 90 48 8d 4d b8 e8 46 58 c3 fc 90', '0F B6 C3', 'B0 01 90'),
    # ("2020.2.0",  '48 8d 4d 28 e8 30 72 c4 fc 90 48 8d 4d b8 e8 26 72 c4 fc 90', '0F B6 C3', 'B0 01 90'),
    # ("2020.1.17", '48 8d 4d 30 e8 e2 b9 11 fd 90 48 8d 4d b8 e8 d8 b9 11 fd 90', '0F B6 C3', 'B0 01 90'),
    # ("2020.1.10", '48 8d 4d 30 e8 92 64 13 fd 90 48 8d 4d b8 e8 88 64 13 fd 90', '0F B6 C3', 'B0 01 90'),
    # ("2020.1.0",  '48 8d 4d 30 e8 f2 9d 15 fd 90 48 8d 4d b8 e8 e8 9d 15 fd 90', '0F B6 C3', 'B0 01 90'),
    # ("2019.4.32", '48 8d 4d 30 e8 2f 9b 75 fe 90 48 8d 4d e8 e8 25 9b 75 fe 90', '0F B6 C3', 'B0 01 90'),
    # ("2019.x-2020.x", '48 8d 4d 0b001..000 e8 . . . 0b11111... 90 48 8d 4d 0b1.1.1000 e8 . . . 0b11111... 90', '0F B6 C3', 'B0 01 90'),
    # ("2018.4.36", '48 83 7d 30 00 76 0a 48 8d 55 48 e8 81 75 4b fe 90', '0F B6 C3', 'B0 01 90'),
    # ("2018.2.0",  '48 83 7d 30 00 76 0a 48 8d 55 48 e8 27 b0 4d fe 90', '0F B6 C3', 'B0 01 90'),
    # ("2018.1.0",  '48 83 7d 30 00 76 0a 48 8d 55 48 e8 d5 df 75 ff 90', '0F B6 C3', 'B0 01 90'),
    ("2021.2.7+",  (('48 8B 4C 24 60 E8 10 53 43 fe 90', '0F B6 C3', 'B0 01 90'),)),
    ("2021.x",     (('48 8B 4C 24 60 E8 .  .  .  FE 90', '0F B6 C3', 'B0 01 90'),)),
    ("2020.3.25+", (('48 8d 4d 28 E8 .  .  .  FE 90 48 8D 4D B8 E8 .  .  .  FE 90', '0F B6 C3', 'B0 01 90'),
                    ('A1 64 00 00 33 FF 40 84 F6', '0F 84', '90 E9'),)),
    ("2020.1",     (('48 8B CB E8 DA 2D 00 00 84 C0', '75', 'EB'),)),
    ("2020.2-3",   (('48 8B CB E8 8D 34 00 00 84 C0', '75', 'EB'),)),
)

default_path = r"C:\Program Files\Unity\Hub\Editor"


def main(arg):
    os.system("title GENERIC UNITY 2020-21 PATCHER")
    print("### GENERIC UNITY 2020-21 PATCHER ###")
    try:
        if not os.path.isfile(arg):
            if os.path.isfile(os.path.abspath("Unity.exe")):
                arg = os.path.abspath("Unity.exe")
            elif os.path.isdir(default_path):
                candidates = sorted(os.listdir(default_path), reverse=True)
                for candidate in candidates:
                    candidate = os.path.join(default_path, candidate, "Editor/Unity.exe")
                    if os.path.isfile(candidate):
                        arg = candidate
                        break
        if not arg:
            raise FileNotFoundError("Couldn't find a valid unity installation.")
        print(f'patching "{arg}"...')
        with open(arg, "r+b") as f:
            data = f.read()
            for ver, patterns in patches:
                patch = Patch(ver, patterns)
                if patch.prepare(data):
                    print(f"found matching patch for unity {patch.ver}")
                    if not os.path.isfile(arg + ".bak"):
                        shutil.copyfile(arg, arg + ".bak")
                    patch.apply(f)
                    if patch.disable_licensing:
                        lic_path = os.path.join(os.path.dirname(arg), "Data/Resources/Licensing")
                        if os.path.isdir(lic_path):
                            print("disabling licensing...", end="")
                            os.rename(lic_path, os.path.join(os.path.dirname(lic_path), "Licensing_disabled"))
                            print(" done")
                    print("done")
                    break
            else:
                print("this unity version is not supported")
    except PermissionError as e:
        print(e)
        if not is_user_admin():
            print("re-trying with admin permissions...")
            time.sleep(2)
            run_as_admin()
            return
    
    input("press any key to exit")


class Sig:
    def __init__(self, sig, orig, patch):
        self.orig, self.orig_bits = self.parse_sig(orig)
        self.orig: bytes = self.escape_sig(self.orig)
        self.patch: str = patch
        self.sig, self.sig_bits = self.parse_sig(sig)
        self.length = len(bytes.fromhex(self.sig.replace(".", "00").replace("?", "00")))
        self.sig: bytes = self.escape_sig(self.sig)
        self.found_offsets: list[int] = []

    @staticmethod
    def parse_sig(sig) -> tuple[str, dict[int, tuple[int, int]]]:
        bit_sigs = {}
        sig_list = sig.split(" ")
        for i, ele in enumerate(sig_list[:]):
            if re.fullmatch(r"0b[\d.?]{8}", ele):
                match_mask = 0
                ignore_mask = 0b11111111
                for k, c in enumerate(ele[2:]):
                    if c == "." or c == "?":
                        ignore_mask -= 1 << (7 - k)
                    elif c == "1":
                        match_mask += 1 << (7 - k)

                bit_sigs[i] = (match_mask, ignore_mask)
                sig_list[i] = "."
        return ' '.join(sig_list), bit_sigs

    @staticmethod
    def escape_sig(sig) -> bytes:
        return b".".join(map(re.escape, map(bytes.fromhex, sig.split("."))))

    @property
    def offset(self):
        if not self.found_offsets:
            raise Exception("No offsets")
        return self.found_offsets[0]

    def get_match_offsets(self, data) -> list[int]:
        for m in re.finditer(self.sig + self.orig, data):
            for i, ele in enumerate(m[0].split(b" ")):
                if i in self.sig_bits:
                    if (int(ele, 16) & self.sig_bits[i][1]) != self.sig_bits[i][0]:
                        break
            else:
                self.found_offsets.append(m.start())
        return self.found_offsets

    def exists_patched(self, data) -> bool:
        return bool(re.search(self.sig + self.escape_sig(self.patch), data))


class Patch:
    def __init__(self, ver: str, patch_sigs: tuple[tuple[str, str, str], ...], disable_licensing: bool = True):
        self.ver: str = ver
        self.sigs: list[Sig] = [Sig(*patch) for patch in patch_sigs]
        self.sigs_to_apply: list[Sig] = []
        self.disable_licensing = disable_licensing

    def prepare(self, data) -> bool:
        for sig in self.sigs:
            offsets = sig.get_match_offsets(data)
            if len(offsets) == 1:
                self.sigs_to_apply.append(sig)
            elif not offsets:
                if not sig.exists_patched(data):
                    return False
                print(f"Signature {sig.sig.hex()} of {self.ver} already patched")
            elif len(offsets) > 1:
                print(f"Signature for {self.ver} is ambiguous (offsets: {hex(offsets[0])}, {hex(offsets[1])})")
                return False
        return True

    def apply(self, f):
        for sig in sorted(self.sigs_to_apply, key=operator.attrgetter("offset")):
            print(f"patching {sig.orig.hex()} to {sig.patch.replace(' ', '').lower()} at offset {hex(sig.offset + sig.length)}")
            f.seek(sig.offset + sig.length)
            f.write(bytes.fromhex(sig.patch))


def is_user_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def run_as_admin():
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable,
                                        " ".join(f'"{x}"' for x in sys.argv), os.getcwd(), 1)


if __name__ == "__main__":
    main("".join(sys.argv[1:]))
