//date: 2022-09-15T17:20:18Z
//url: https://api.github.com/gists/138861a9cfea5beeeb8a61dc1d70841d
//owner: https://api.github.com/users/ryandotsmith

package ens

import (
    "strings"

    "github.com/ethereum/go-ethereum/common"
    "github.com/ethereum/go-ethereum/crypto"
)

func Namehash(name string) common.Hash {
    var (
        h      common.Hash
        labels = strings.Split(name, ".")
    )
    for i := len(labels) - 1; i >= 0; i-- {
        h = crypto.Keccak256Hash(
            h.Bytes(),
            crypto.Keccak256Hash([]byte(labels[i])).Bytes(),
        )
    }
    return h
}