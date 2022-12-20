//date: 2022-12-20T16:50:14Z
//url: https://api.github.com/gists/fb65eaf808d02044a5edd7508b0f7188
//owner: https://api.github.com/users/lazyfrosch

package net

import (
	"encoding/binary"
	"fmt"
	"net"
)

// GetNetworkRange returns begin and end address for hosts in a network.
//
// This range excludes network, gateway (first address) and broadcast address.
//
// Example: 10.0.0.0/23 would be 10.0.0.2 - 10.0.1.254
func GetNetworkRange(cidr string) (*net.IP, *net.IP, error) {
	_, network, err := net.ParseCIDR(cidr)
	if err != nil {
		return nil, nil, fmt.Errorf("could not parse CIDR: %w", err)
	}

	// Convert byte arrays to uint values to apply bit logic
	// Note: V4 only?
	uNetwork := binary.BigEndian.Uint32(network.IP)
	uMask := binary.BigEndian.Uint32(network.Mask)
	uBroadcast := (uNetwork & uMask) | (uMask ^ 0xffffffff)

	first := make(net.IP, 4)
	binary.BigEndian.PutUint32(first, uNetwork+2)

	last := make(net.IP, 4)
	binary.BigEndian.PutUint32(last, uBroadcast-1)

	return &first, &last, nil
}
