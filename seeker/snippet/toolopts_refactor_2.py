#date: 2024-06-14T17:08:08Z
#url: https://api.github.com/gists/ce4f37395f7efaffd7cc0c5c8af767fe
#owner: https://api.github.com/users/r0x0d

import copy
from argparse import Namespace
import pprint


class BaseConfig:
    debug = False
    username = None
    config_file = None
    password = "**********"
    no_rhsm = False
    enablerepo = []
    disablerepo = []
    pool = None
    rhsm_hostname = None
    rhsm_port = None
    rhsm_prefix = None
    autoaccept = None
    auto_attach = None
    restart = None
    activation_key = None
    org = None
    arch = None
    no_rpm_va = False
    eus = False
    els = False
    activity = None

    # Settings
    incomplete_rollback = None
    tainted_kernel_module_check_skip = None
    outdated_package_check_skip = None
    allow_older_version = None
    allow_unavailable_kmods = None
    configure_host_metering = None
    skip_kernel_currency_check = None

    def set_opts(self, opts):
        """Set ToolOpts data using dict with values from config file.
        :param opts: Supported options in config file
        """
        for key, value in opts.items():
            if value and hasattr(BaseConfig, key):
                setattr(BaseConfig, key, value)

class FileConfig(BaseConfig):
    def __init__(self, config_files):
        self._config_files = config_files

    def run(self):
        opts = self.options_from_config_files()
        self.set_opts(opts)

    mock_data = {'username': 'test', 'allow_incomplete_rollback': '1'}

    def options_from_config_files(self):
        unparsed_opts = {}
        for key, value in self.mock_data.items():
            unparsed_opts[key] = value

        return unparsed_opts

class CliConfig(BaseConfig):
    def __init__(self, opts):
        self._opts = vars(opts)

    def run(self):
        opts = self._normalize_opts(self._opts)
        self.set_opts(opts)
    
    def _normalize_opts(self, opts):
        unparsed_opts = copy.copy(opts)
        unparsed_opts["activation_key"] = opts.pop("activationkey", None)
        unparsed_opts["auto_accept"] = opts.pop("y", False)
        unparsed_opts["activity"] = opts.pop("command", "convert")

        return unparsed_opts

class ToolOpts(BaseConfig):
    def __init__(self, config_sources):
        super(ToolOpts, self).__init__()
        for config in reversed(config_sources):
            config.run()

# -------- execution
def initialize_toolopts(config_sources):
    return ToolOpts(config_sources=config_sources)

# parsed_opts is the output of the CLI parsed from argparse
parsed_opts = Namespace(
    activationkey=None, auto_attach=False, command='analyze', config_file=None, 
    debug=True, disablerepo=None, els=False, enablerepo=None, eus=False, 
    no_rhsm= "**********"=False, org=None, password='c2r_pswd2', 
    pool='2c9435a18dcbf599018e0db3afc547f3', restart=False, 
    serverurl='subscription.rhsm.stage.redhat.com', username='c2r_main2', 
    y=True
)

# FileConfig takes highest prioirty here
tool_opts = ToolOpts(config_sources=(CliConfig(parsed_opts), FileConfig(("test.ini",))))
pprint.pprint(tool_opts.username)

# CLIConfig takes highest prioirty here
tool_opts = ToolOpts(config_sources=(FileConfig(("test.ini",)),CliConfig(parsed_opts)))
pprint.pprint(tool_opts.username)