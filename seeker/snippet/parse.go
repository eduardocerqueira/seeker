//date: 2021-10-20T17:15:30Z
//url: https://api.github.com/gists/80f6d8727dd0dad5b35620aedea0ae4e
//owner: https://api.github.com/users/nddeluca

package main

import (
	"encoding/base64"
	"fmt"
	"log"
	"os"

	sdk "github.com/cosmos/cosmos-sdk/types"
	auth "github.com/cosmos/cosmos-sdk/x/auth"
	kava "github.com/kava-labs/kava/app"
)

func main() {
	//
	// bootstrap kava chain config for decoding addresses
	// with kava bech32 prefix
	//
	kavaConfig := sdk.GetConfig()
	kava.SetBech32AddressPrefixes(kavaConfig)
	kava.SetBip44CoinType(kavaConfig)
	kavaConfig.Seal()

	//
	// codec for decoding amino and json
	//
	cdc := kava.MakeCodec()

	//
	// Base64 encoded bytes of length prefixed amino
	//
	txBase64 := os.Args[1]

	//
	// decode base64 to bytes
	//
	decoded, err := base64.StdEncoding.DecodeString(txBase64)
	if err != nil {
		log.Fatal(err)
		os.Exit(1)
	}

	//
	// decode length prefixed amino
	//
	var decodeTx auth.StdTx
	err = cdc.UnmarshalBinaryLengthPrefixed(decoded, &decodeTx)
	if err != nil {
		log.Fatal(err)
		os.Exit(1)
	}

	//
	// encode to json
	//
	jsonData, err := cdc.MarshalJSON(decodeTx)
	if err != nil {
		log.Fatal(err)
		os.Exit(1)
	}

	//
	// print json
	//
	fmt.Println(string(jsonData))
}