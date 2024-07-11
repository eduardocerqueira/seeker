//date: 2024-07-11T16:42:54Z
//url: https://api.github.com/gists/9e9eff0e622982a82a672a20beb08e66
//owner: https://api.github.com/users/b4nst

import (
	"net"
	"net/netip"
)

func PrefixToIPNet(p netip.Prefix) net.IPNet {
	ip := p.Masked().Addr().AsSlice()
	return net.IPNet{
		IP:   ip,
		Mask: net.CIDRMask(p.Bits(), len(ip)*8),
	}
}