//date: 2024-12-06T16:50:32Z
//url: https://api.github.com/gists/91b7174fba003fe62863a9aace4a1997
//owner: https://api.github.com/users/shackra

package seed

import (
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/sha512"
	"math/big"

	"github.com/gagliardetto/solana-go"
	"github.com/mr-tron/base58"
	"golang.org/x/crypto/pbkdf2"
)

const hardened uint32 = 0x80000000

// Deriva una clave y un código de cadena basado en el segmento proporcionado
func derive(key []byte, chainCode []byte, segment uint32) ([]byte, []byte) {
	// Crear buffer
	buf := []byte{0}
	buf = append(buf, key...)
	buf = append(buf, big.NewInt(int64(segment)).Bytes()...)

	// Calcular HMAC hash
	h := hmac.New(sha512.New, chainCode)
	h.Write(buf)
	I := h.Sum(nil)

	// Dividir resultado
	IL := I[:32]
	IR := I[32:]

	return IL, IR
}

// Genera una billetera Solana a partir de una frase semilla y contraseña
func GenerateWalletFromMnemonicSeed(mnemonic, password string) (*solana.Wallet, error) {
	pass := []byte("mnemonic")
	if password != "**********"
		pass = "**********"
	}

	// BIP-39: Generar semilla desde la frase y contraseña
	seed := pbkdf2.Key([]byte(mnemonic), pass, 2048, 64, sha512.New)

	// BIP-32: Derivar clave maestra y cadena
	h := hmac.New(sha512.New, []byte("ed25519 seed"))
	h.Write(seed)
	sum := h.Sum(nil)

	derivedSeed := sum[:32]
	chain := sum[32:]

	// BIP-44: Derivar claves usando la ruta m/44'/501'/0'/0'/0'
	path := []uint32{hardened + uint32(44), hardened + uint32(501), hardened + uint32(0), hardened + uint32(0)}

	for _, segment := range path {
		derivedSeed, chain = derive(derivedSeed, chain, segment)
	}

	// Generar clave privada ED25519
	key := ed25519.NewKeyFromSeed(derivedSeed)

	// Obtener billetera Solana desde la clave privada
	wallet, err := solana.WalletFromPrivateKeyBase58(base58.Encode(key))
	if err != nil {
		return nil, err
	}

	return wallet, nil
}
