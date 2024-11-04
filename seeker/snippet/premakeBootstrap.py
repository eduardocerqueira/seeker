#date: 2024-11-04T16:53:17Z
#url: https://api.github.com/gists/1507ccef3f71d8442b05d58c239d6a6b
#owner: https://api.github.com/users/lolrobbe2

import subprocess
import os
import sys

# Define the path to vswhere.exe
VSWHERE_PATH = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"


def run_nmake_with_vsdevcmd(vs_path, vs_version, architecture="64"):
    """Run nmake within the Visual Studio environment set up by vsdevcmd.bat."""
    vsdevcmd_path = os.path.join(vs_path, "Common7", "Tools", "VsDevCmd.bat")
    nmake_command = f"nmake MSDEV={vs_version} -f Bootstrap.mak windows"

    if os.path.exists(vsdevcmd_path):
        # Run both vsdevcmd.bat and nmake in a single command
        command = f'call "{vsdevcmd_path}" && {nmake_command}'
        subprocess.call(command, shell=True)
    else:
        print("Could not find vsdevcmd.bat to setup Visual Studio environment.")
        exit(2)


def legacy_visual_bootstrap(vs_version, version_no_point):
    """Bootstrap legacy Visual Studio versions."""
    vs_env_var = f"VS{version_no_point}COMNTOOLS"
    vs_path = os.getenv(vs_env_var)

    if not vs_path or not os.path.exists(os.path.join(vs_path, "VsDevCmd.bat")):
        print("Could not find vsdevcmd.bat to setup Visual Studio environment.")
        exit(2)

    run_nmake_with_vsdevcmd(vs_path, vs_version)


def vswhere_visual_bootstrap(vs_version, version_min, version_max):
    """Bootstrap Visual Studio versions using vswhere."""
    if not os.path.exists(VSWHERE_PATH):
        print("Could not find vswhere.exe.")
        exit(2)

    cmd = f'"{VSWHERE_PATH}" -nologo -latest -version [{version_min},{version_max}) -property installationPath'
    installation_path = subprocess.check_output(cmd, shell=True, text=True).strip()

    if os.path.exists(installation_path):
        run_nmake_with_vsdevcmd(installation_path, vs_version)
    else:
        print("Could not find a valid installation path for Visual Studio.")
        exit(2)


def bootstrap_latest():
    """Bootstrap the latest version of Visual Studio."""
    if os.path.exists(VSWHERE_PATH):
        cmd = f'"{VSWHERE_PATH}" -nologo -latest -property catalog.productLineVersion'
        latest_version = subprocess.check_output(cmd, shell=True, text=True).strip()
        legacy_visual_bootstrap(f"vs{latest_version}", latest_version)
    else:
        print("Could not find vswhere.exe.")
        exit(2)


def main():
    """Main function to handle the Visual Studio version bootstrap process."""
    if len(sys.argv) < 2:
        bootstrap_latest()
        return

    vsversion = sys.argv[1]

    if vsversion == "vs2010":
        legacy_visual_bootstrap(vsversion, "100")
    elif vsversion == "vs2012":
        legacy_visual_bootstrap(vsversion, "110")
    elif vsversion == "vs2013":
        legacy_visual_bootstrap(vsversion, "120")
    elif vsversion == "vs2015":
        legacy_visual_bootstrap(vsversion, "140")
    elif vsversion == "vs2017":
        vswhere_visual_bootstrap(vsversion, "15.0", "16.0")
    elif vsversion == "vs2019":
        vswhere_visual_bootstrap(vsversion, "16.0", "17.0")
    elif vsversion == "vs2022":
        vswhere_visual_bootstrap(vsversion, "17.0", "18.0")
    else:
        print(f"Unrecognized Visual Studio version {vsversion}.")
        exit(2)


if __name__ == "__main__":
    main()
