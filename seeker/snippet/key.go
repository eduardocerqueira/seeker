//date: 2022-09-20T17:17:39Z
//url: https://api.github.com/gists/38c1ee50126e5a859a99beb184bdf753
//owner: https://api.github.com/users/marbar3778

package key

import (
	"bytes"
	"strings"

	"golang.org/x/exp/constraints"

	"github.com/axelarnetwork/utils/convert"
)

// DefaultDelimiter represents the default delimiter used for the KV store keys when concatenating them together
const DefaultDelimiter = "_"

// Key provides a type safe way to interact with the store
type Key interface {
	Append(Key) Key
	Bytes(delimiter ...string) []byte
}

type basicKey struct {
	particles [][]byte
}

func (k basicKey) Append(suffix Key) Key {
	var bz [][]byte
	switch suffix := suffix.(type) {
	case basicKey:
		bz = suffix.particles
	default:
		bz = [][]byte{suffix.Bytes()}
	}
	return basicKey{
		particles: append(k.particles, bz...),
	}
}

func (k basicKey) Bytes(delimiter ...string) []byte {
	del := DefaultDelimiter
	if len(delimiter) == 1 {
		del = delimiter[0]
	}
	return bytes.Join(k.particles, []byte(del))
}

// FromBz creates a new Key from bytes
func FromBz(key []byte) Key {
	return &basicKey{particles: [][]byte{key}}
}

// FromStr creates a new Key from a string
func FromStr(key string) Key {
	return FromBz([]byte(strings.ToLower(key)))
}

// FromUInt creates a new Key from any unsigned integer type
func FromUInt[T constraints.Unsigned](key T) Key {
	return FromBz(convert.IntToBytes(key))
}

// FromUInt creates a new Key from any unsigned integer type
func FromInt[T constraints.Integer](key T) Key {
	return FromBz(convert.IntToBytes(key))
}