#date: 2024-12-16T17:06:13Z
#url: https://api.github.com/gists/6acdbd7d97179837d348b05101b412f2
#owner: https://api.github.com/users/gasinvein

from dns import resolver, reversename
from dns.exception import DNSException

from ipapython import dnsutil
import ipaserver.plugins.cert


def _ip_ptr_records_dns(ip):
    """
    Look up PTR record(s) for IP address.

    :return: a ``set`` of IP addresses, possibly empty.

    """
    rname = dnsutil.DNSName(reversename.from_address(ip))
    try:
        answer = resolver.resolve(rname, 'PTR')
    except DNSException:
        ptrs = set()
    else:
        ptrs = {r.to_text() for r in answer.rrset}
    return ptrs


_ip_ptr_records_ldap = ipaserver.plugins.cert._ip_ptr_records
ipaserver.plugins.cert._ip_ptr_records = _ip_ptr_records_dns