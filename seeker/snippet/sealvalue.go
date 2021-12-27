//date: 2021-12-27T16:44:03Z
//url: https://api.github.com/gists/43534158d7f6a4886d9da19fa76dae2d
//owner: https://api.github.com/users/bencord0

package main

import (
	"encoding/base64"
	"fmt"
	"golang.org/x/crypto/nacl/box"
)

func main() {
	plaintext := "hello"
  
  # retrieve this from data.github_actions_public_key.key
	publicKey := ""

	encrypted, err := encrypt(plaintext, publicKey)
	if err != nil {
		fmt.Errorf("failed to encrypt: %v", err)
		return
	}

	encoded := encode(encrypted)
	fmt.Println(encoded)
}

func encrypt(plaintext, encodedPublicKey string) ([]byte, error) {
	publicKeyBytes, err := base64.StdEncoding.DecodeString(encodedPublicKey)
	if err != nil {
		return nil, err
	}

	var publicKeyBytes32 [32]byte
	copiedLen := copy(publicKeyBytes32[:], publicKeyBytes)
	if copiedLen == 0 {
		return nil, fmt.Errorf("could not convert publicKey to bytes")
	}

	plaintextBytes := []byte(plaintext)
	var encryptedBytes []byte

	cipherText, err := box.SealAnonymous(encryptedBytes, plaintextBytes, &publicKeyBytes32, nil)
	if err != nil {
		return nil, err
	}

	return cipherText, nil
}

func encode(unencoded []byte) string {
 	return base64.StdEncoding.EncodeToString(unencoded)
}