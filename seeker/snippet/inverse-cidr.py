#date: 2021-10-29T16:48:03Z
#url: https://api.github.com/gists/2726d4ac82cbefa474a04ffd6523e023
#owner: https://api.github.com/users/ColdHeat

"""Use it like this: main('192.168.1.0/24')"""

IPV4_MIN = 0
IPV4_MAX = 0xFFFFFFFF

def not_network(ipv4_address, ipv4_netmask):
    assert IPV4_MIN <= ipv4_address <= IPV4_MAX
    assert IPV4_MIN <= ipv4_netmask <= IPV4_MAX

    def hostmask_netmask(m):
        """Convert hostmask to netmask and viceversa"""
        return ~m & IPV4_MAX

    ipv4_hostmask = hostmask_netmask(ipv4_netmask)
    ipv4_first_ip = ipv4_address
    ipv4_last_ip = ipv4_address | ipv4_hostmask

    cidrs = []

    # Find all CIDRs before the very first IP of the specified network.
    address = IPV4_MIN
    hostmask = IPV4_MAX
    while address != ipv4_first_ip:
        if address | hostmask < ipv4_first_ip:
            # largest possible network that starts at *address* before *ipv4_first_ip*
            cidrs.append((address, hostmask_netmask(hostmask)))
            address = (address | hostmask) + 1
        hostmask >>= 1

    # Find  all CIDRs after the very last IP of the specified network.
    # The algorithm as the same as above but we look in reverse direction
    address = IPV4_MAX
    hostmask = IPV4_MAX
    while address != ipv4_last_ip:
        if address ^ hostmask > ipv4_last_ip:
            # largest possible network that starts at *address ^ hostmask* after *ipv4_last_ip*
            cidrs.append((address ^ hostmask, hostmask_netmask(hostmask)))
            address = (address ^ hostmask) - 1
        hostmask >>= 1

    return sorted(cidrs)


def not_network_str(ipv4_network):
    from ipaddress import IPv4Network
    n = IPv4Network(ipv4_network)
    return not_network(int(n.network_address), int(n.netmask))


def main(ipv4_network):
    from ipaddress import IPv4Address, IPv4Network
    n = IPv4Network(ipv4_network)
    for i, m in not_network(int(n.network_address), int(n.netmask)):
        print(IPv4Network("{}/{}".format(IPv4Address(i), IPv4Address(m))))