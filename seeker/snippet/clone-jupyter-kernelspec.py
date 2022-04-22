#date: 2022-04-22T17:01:22Z
#url: https://api.github.com/gists/05a48e225574f122795d5586c9bb2ec4
#owner: https://api.github.com/users/rcthomas

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
import shutil
import sys
from textwrap import dedent

from jupyter_client.kernelspec import _list_kernels_in, jupyter_data_dir

def main():
    path = Path(sys.prefix) / "share/jupyter/kernels"
    source_kernelspecs = get_source_kernelspecs(path)
    args = parse_arguments(source_kernelspecs)

    new_name = args.new_name or args.kernel_name

    src_resource_dir = Path(source_kernelspecs[args.kernel_name])
    new_resource_dir = Path(jupyter_data_dir()) / new_name

    print(f"Copying\n  '{src_resource_dir}'\nto\n  '{new_resource_dir}'")
    if not args.yes:
        confirm = input(f"ARE YOU SURE [y/N]? ")
        if not confirm.startswith("y"):
            print("OK, not copying then!")
            sys.exit(0)

    try:
        shutil.rmtree(new_resource_dir)
    except FileNotFoundError:
        pass

    shutil.copytree(src_resource_dir, new_resource_dir)

def get_source_kernelspecs(path):
    source_kernelspecs = _list_kernels_in(path)
    assert source_kernelspecs, f"No kernelspecs, searched {path}"
    return source_kernelspecs

def parse_arguments(source_kernelspecs):

    description = dedent("""\
    Copy a sys.prefix kernel resource directory to a user kernel resource
    directory by name.  Any existing user resource directory by that name
    will be removed and replaced.
    """)

    parser = ArgumentParser(
        description=description,
        epilog=format_source_kernelspecs(source_kernelspecs),
        formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "kernel_name",
        help="Name of the kernel-spec you want to clone, list below.",
        choices=source_kernelspecs.keys(),
        metavar="{kernel-name}"
    )
    parser.add_argument(
        "--new-name", "-n",
        help="Optional new name for the cloned kernel-spec."
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Prevent the 'are you sure' confirmation prompt.",
    )
    return parser.parse_args()

def format_source_kernelspecs(source_kernelspecs):
    kernel_name_label = "kernel-name:"
    resource_dir_label = "resource-dir:"

    len_kernel_name_label = len(kernel_name_label)
    len_resource_dir_label = len(resource_dir_label)

    width = len_kernel_name_label
    for name in source_kernelspecs:
        if (len_name := len(name)) > width:
            width = len_name

    text = "kernel-name (and source directory) choices:\n"
    for name in sorted(source_kernelspecs):
        text += f"  {name:<{width}} ({source_kernelspecs[name]})\n"

    text += "\nsimply clone a kernel-spec:\n\n"
    text += f"    $ clone-jupyter-kernel {name}\n\n"
    text += f"  This will clone the kernel-spec '{name}' from\n"
    text += f"    {source_kernelspecs[name]}\n"
    text += f"  to\n"
    text += f"    {jupyter_data_dir()}/{name}\n\n"

    text += "\nclone a kernel-spec, but give it a new name too:\n\n"
    text += f"    $ clone-jupyter-kernel {name} --new-name=mykernel\n\n"
    text += f"  This will clone the kernel-spec '{name}' from\n"
    text += f"    {source_kernelspecs[name]}\n"
    text += f"  to\n"
    text += f"    {jupyter_data_dir()}/mykernel\n"

    return text

if __name__ == "__main__":
    main()