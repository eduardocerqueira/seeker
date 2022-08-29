#date: 2022-08-29T17:17:52Z
#url: https://api.github.com/gists/c621ca8208ed4d983b3cdc20e0464256
#owner: https://api.github.com/users/hkrutzer

from datadog_checks.base import AgentCheck
from datadog_checks.base.utils.subprocess_output import get_subprocess_output

__version__ = "0.0.1"

class AptOutdatedCheck(AgentCheck):
    def check(self, instance):
        stdout, package_counts, retcode = get_subprocess_output("/usr/lib/update-notifier/apt-check", self.log, raise_on_empty_output=False)
        outdated_total, outdated_security = package_counts.split(';')
        self.gauge("apt.package.outdated.count", int(outdated_total), tags=['kind:total'] + self.instance.get('tags', []))
        self.gauge("apt.package.outdated.count", int(outdated_security), tags=['kind:security'] + self.instance.get('tags', []))