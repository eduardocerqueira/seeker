#date: 2023-01-06T16:32:23Z
#url: https://api.github.com/gists/9824bd60891dc47b0631565608783df6
#owner: https://api.github.com/users/markjoshwel

"""
a gentoo install script
=======================

warning: this script does not do the following (by default)

 - check if your shell is running as root
 - partition your disks
 - ensure you have the correct time and date

do these before running, if you haven't adjusted the script to
account for these.

----------------------------------------------------------------

notes:

 - change MIRROR, COMMAND_SETS, DEFAULT_SETS and FILES as you
   see fit
 - for changing of directories, use the custom directive
   '!wd:<path>', as cd doesnt work with subprocess.run, see set
   'dlstage3' for an example usage.
   the directory change will persist for the entirety of the SET
   ONLY.
 - file content (make.conf, etc) is stored at the end of the file
"""

from argparse import ArgumentParser, RawTextHelpFormatter
from subprocess import run, CalledProcessError, STDOUT
from textwrap import dedent
from time import sleep

MIRROR = "https://download.nus.edu.sg/mirror/gentoo/"  # must end with '/'!

COMMAND_SETS: dict[str, list[str]] = {
    "diskprep": [
        # disk preperation
        "mkfs.vfat -F 32 /dev/nvme0n1p1",  #  boot (mbr/efi)
        "mkfs.btrfs -f /dev/nvme0n1p5",  #    linux root
        "mkswap /dev/nvme0n1p6",  #           swap
        "mkfs.btrfs -f /dev/nvme0n1p7",  #    other data
        "swapon /dev/nvme0n1p6",
        "mkdir --parents /mnt/gentoo",
        "mount /dev/nvme0n1p5 /mnt/gentoo",
    ],
    "dlstage3": [
        # downloading stage3
        "!wd:/mnt/gentoo",
        f"wget \"{MIRROR}releases/amd64/autobuilds/$(wget -qO- {MIRROR}releases/amd64/autobuilds/latest-stage3-amd64-musl-llvm.txt | tail -1 | cut -d ' ' -f 1)\"",
        "tar xpvf stage3-*.tar.xz --xattrs-include='*.*' --numeric-owner",
        "rm stage3-*.tar.xz",
    ],
    "instbase": [
        # installation and configuration to make a 'mini' stage 4/ms4
        # (minimal install w/o bootloader)
    ],
    "pkgms4": [
        # packaging the mini stage4/ms4 for use elsewhere
    ],
    "instfull": [
        # installation and configuration for a full user installation
        # (from a minimal install to a ready-to-use install)
    ],
}

DEFAULT_SETS = ["instbase", "instfull"]


def start(
    sets: list[str] = DEFAULT_SETS,
    exclude_sets: list[str] = [],
    dry: bool = False,
):
    """
    start the script

    sets: list[str] = DEFAULT_SETS
        sets to run here using their variable names
    exclude_sets: list[str] = []
        sets to exclude from the original sets argument
    dry: bool = False
        when enabled, does not run any command
    """
    if not dry:
        for s in range(3):
            print(f"\rrunning the install script in {3-s}", end="")
            sleep(1)

        print("\rrunning the install script now ")

    for cs in COMMAND_SETS:
        if (cs not in sets) or (cs in exclude_sets):
            continue

        print(f"\n----- running set '{cs}' -----\n")

        cwd: str | None = None
        for cmd in COMMAND_SETS[cs]:
            print(f"--> {cmd}")

            # match for custom directives (i.e. '!wd:/mnt/gentoo')
            match cmd.split("!"):
                case ["", directive]:  # matched format '!x'
                    match directive.split(":"):
                        case ["wd", directory]:
                            cwd = directory
                        case _:
                            print(f"unsupported directive: '{cmd}'")
                            exit(2)

                case _:  # is not special directive
                    if dry:
                        continue

                    retcode: int = 0
                    try:
                        retcode = run(
                            cmd,
                            shell=True,
                            check=True,
                            stderr=STDOUT,
                            cwd=cwd,
                        ).returncode
                    except CalledProcessError as e:
                        print(f"\n{e.__class__.__name__}: {e}")
                        exit(retcode)

            if not dry:
                print()


def parse_args() -> tuple[list[str], list[str], bool]:
    parser = ArgumentParser(
        description="a gentoo install script", formatter_class=RawTextHelpFormatter
    )

    parser.add_argument(
        "sets",
        help=(
            f"comma-seperated sets to run, defaults to '{','.join([cs for cs in DEFAULT_SETS])}'"
            f"\navailable sets: {' '.join([cs for cs in COMMAND_SETS])}"
        ),
        nargs="?",
        type=str,
        default=",".join([cs for cs in DEFAULT_SETS]),
    )

    parser.add_argument(
        "-e",
        "--exclude_sets",
        help=(f"comma-seperated sets to exclude , defaults to ''"),
        type=str,
        default="",
    )

    parser.add_argument(
        "-d",
        "--dry",
        help="state commands but not execute them",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-p",
        "--print-file",
        help=(
            "print a stored file"
            "\nif given, no sets will execute. file contents are printed and the script exits."
        ),
        type=str,
        default="",
    )

    args = parser.parse_args()

    if args.print_file != "":
        if args.print_file not in FILES:
            print(f"error: file '{args.print_file}' is not a stored file")
            exit(1)

        print(FILES[args.print_file])

    arg_sets: list[str] = [""]

    match args.sets:
        case "sets":
            print(f"available sets: {' '.join([cs for cs in COMMAND_SETS])}")
            exit(0)

        case "all":
            arg_sets = [cs for cs in COMMAND_SETS]

        case _:
            arg_sets = args.sets.split(",")
            for as_ in arg_sets:
                if as_ not in [cs for cs in COMMAND_SETS]:
                    print(
                        f"error: unknown set '{as_}'"
                        f"\navailable sets: {' '.join([cs for cs in COMMAND_SETS])}"
                    )
                    exit(1)

    exclude_sets: list[str] = []

    exclude_sets = args.exclude_sets.split(",")
    for as_ in arg_sets:
        if as_ not in [cs for cs in COMMAND_SETS]:
            print(
                f"error: unknown exclude set '{as_}'"
                f"\navailable sets: {' '.join([cs for cs in COMMAND_SETS])}"
            )
            exit(1)

    return arg_sets, exclude_sets, args.dry


FILES: dict[str, str] = {
    "make.conf": """# this stage was built with the bindist use flag enabled
# see /usr/share/portage/config/make.conf.example
COMMON_FLAGS="-march=native -O2 pipe"
CFLAGS="${COMMON_FLAGS}"
CXXFLAGS="${COMMON_FLAGS}"
FCFFLAGS="${COMMON_FLAGS}"
FFFLAGS="${COMMON_FLAGS}"
LC_MESSAGES=C

CHOST="x86_64-gentoo-linux-musl"
GENTOO_MIRRORS="https://download.nus.edu.sg/mirror/gentoo/"
ACCEPT_LICENSE="-* @FREE @BINARY-REDISTRIBUTABLE"

CPU_FLAGS_X86="aes avx avx2 f16c fma3 mmx mmxext pclmul popcnt rdrand sha sse sse2 sse3 sse4_1 sse4_2 sse4a ssse3"
VIDEO_CARDS="amdgpu radeonsi"

MAKEOPTS="-j10"
PORTAGE_NICENESS="10"

USE="wifi bluetooth acpi btrfs\""""
}


if __name__ == "__main__":
    start(*parse_args())