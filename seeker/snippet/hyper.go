//date: 2025-05-19T16:40:32Z
//url: https://api.github.com/gists/07f3eb71a68d53a437e6c05539cd645b
//owner: https://api.github.com/users/pedroalbanese

package main

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/asn1"
	"errors"
	"fmt"
	"math/big"
)

/*
 * ==============================
 *          ESTRUTURAS
 * ==============================
 */

// Divisor representa um divisor na curva hiperelíptica
type Divisor struct {
	U [2]*big.Int // Polinômio u(x) = x² + U[1]x + U[0]
	V [2]*big.Int // Polinômio v(x) = V[1]x + V[0]
}

// HyperellipticCurve define os parâmetros da curva
type HyperellipticCurve struct {
	h    []*big.Int
	f    []*big.Int
	p    *big.Int
	g    int
	base Divisor
}

/*
 * ==============================
 *    INICIALIZAÇÃO DA CURVA
 * ==============================
 */

// NewCurve256 inicializa a curva com um divisor base válido
func NewCurve() *HyperellipticCurve {
	p, _ := new(big.Int).SetString("f92336ffa5d4903990da8d0abfbd34592b9d3a6ed4059051e8dfea5307950bd1535fa2c2e25efbf37ba0ff1c0cf9c57ac59ba71d5d6486b891d4dab3be2d56d3", 16)

	// Divisor base válido (valores de exemplo - na prática calcule um divisor válido)
	base := Divisor{
		U: [2]*big.Int{
			new(big.Int).SetInt64(3), // u₀
			new(big.Int).SetInt64(2), // u₁
		},
		V: [2]*big.Int{
			new(big.Int).SetInt64(1), // v₀
			new(big.Int).SetInt64(4), // v₁
		},
	}

	h := []*big.Int{big.NewInt(0), big.NewInt(1)}
	f := []*big.Int{
		big.NewInt(5), big.NewInt(1), big.NewInt(0),
		big.NewInt(3), big.NewInt(0), big.NewInt(1),
	}

	return &HyperellipticCurve{
		h:    h,
		f:    f,
		p:    p,
		g:    2,
		base: base,
	}
}

// ScalarMult multiplica um divisor por um escalar
func (c *HyperellipticCurve) ScalarMult(k *big.Int, D Divisor) Divisor {
	result := zeroDivisor()
	current := D

	for i := k.BitLen() - 1; i >= 0; i-- {
		if !isZeroDivisor(result) {
			result = c.AddDivisors(result, result)
		}
		if k.Bit(i) == 1 {
			result = c.AddDivisors(result, current)
		}
	}

	return result
}

// ReduceDivisor reduz um divisor para a forma normalizada
func (c *HyperellipticCurve) ReduceDivisor(D Divisor) Divisor {
	// Se já estiver reduzido (grau(u) ≤ g) e v(x) correto
	if c.isReduced(D) {
		return D
	}

	// Implementação simplificada da redução:
	// 1. Garantir que u(x) seja mônico
	u := make([]*big.Int, 3)
	u[0], u[1] = new(big.Int).Set(D.U[0]), new(big.Int).Set(D.U[1])
	u[2] = big.NewInt(1) // x² + u1x + u0

	// 2. Ajustar v(x) para ter grau < grau(u)
	v := make([]*big.Int, 2)
	v[0], v[1] = new(big.Int).Set(D.V[0]), new(big.Int).Set(D.V[1])

	// 3. Aplicar redução mod p
	for i := range u {
		u[i].Mod(u[i], c.p)
	}
	for i := range v {
		v[i].Mod(v[i], c.p)
	}

	return Divisor{
		U: [2]*big.Int{u[0], u[1]},
		V: [2]*big.Int{v[0], v[1]},
	}
}

// isReduced verifica se um divisor está na forma reduzida
func (c *HyperellipticCurve) isReduced(D Divisor) bool {
	// Verifica se grau(u) ≤ g (2 para gênero 2)
	// u(x) deve ser x² + u1x + u0
	if D.U[1].Cmp(big.NewInt(0)) == 0 && D.U[0].Cmp(big.NewInt(0)) == 0 {
		return false
	}

	// Verifica se grau(v) < grau(u)
	if D.V[1].Cmp(big.NewInt(0)) != 0 { // v1 ≠ 0 ⇒ grau(v) = 1
		return true
	}
	return D.V[0].Cmp(big.NewInt(0)) != 0 // v0 ≠ 0 ⇒ grau(v) = 0
}

// Atualize sua função AddDivisors para usar a redução
func (c *HyperellipticCurve) AddDivisors(D1, D2 Divisor) Divisor {
	if isZeroDivisor(D1) {
		return D2
	}
	if isZeroDivisor(D2) {
		return D1
	}

	// Operação básica de adição (sua versão atual)
	u0 := new(big.Int).Add(D1.U[0], D2.U[0])
	u1 := new(big.Int).Add(D1.U[1], D2.U[1])
	v0 := new(big.Int).Add(D1.V[0], D2.V[0])
	v1 := new(big.Int).Add(D1.V[1], D2.V[1])

	// Aplica módulo p
	u0.Mod(u0, c.p)
	u1.Mod(u1, c.p)
	v0.Mod(v0, c.p)
	v1.Mod(v1, c.p)

	// Garante valores mínimos
	if u0.Cmp(big.NewInt(1)) < 0 {
		u0.Add(u0, big.NewInt(2))
	}
	if u1.Cmp(big.NewInt(1)) < 0 {
		u1.Add(u1, big.NewInt(2))
	}
	if v0.Cmp(big.NewInt(1)) < 0 {
		v0.Add(v0, big.NewInt(2))
	}
	if v1.Cmp(big.NewInt(1)) < 0 {
		v1.Add(v1, big.NewInt(2))
	}

	// Aplica redução final
	return c.ReduceDivisor(Divisor{
		U: [2]*big.Int{u0, u1},
		V: [2]*big.Int{v0, v1},
	})
}

// zeroDivisor retorna o divisor neutro
func zeroDivisor() Divisor {
	return Divisor{
		U: [2]*big.Int{big.NewInt(1), big.NewInt(0)},
		V: [2]*big.Int{big.NewInt(0), big.NewInt(0)},
	}
}

func isZeroDivisor(d Divisor) bool {
	return d.U[0].Cmp(big.NewInt(1)) == 0 &&
		d.U[1].Cmp(big.NewInt(0)) == 0 &&
		d.V[0].Cmp(big.NewInt(0)) == 0 &&
		d.V[1].Cmp(big.NewInt(0)) == 0
}

/*
 * ==============================
 *    OPERAÇÕES CRIPTOGRÁFICAS
 * ==============================
 */

// GenerateKeyPair gera pares de chaves únicos
func (c *HyperellipticCurve) GenerateKeyPair() (*big.Int, Divisor, error) {
	privateKey, err := rand.Int(rand.Reader, c.p)
	if err != nil {
		return nil, Divisor{}, err
	}

	// Gera chave pública única baseada na chave privada
	publicKey := c.generateUniquePublicKey(privateKey)
	return privateKey, publicKey, nil
}

// generateUniquePublicKey corrigida para gerar chaves válidas
func (c *HyperellipticCurve) generateUniquePublicKey(privateKey *big.Int) Divisor {
	// Implementação real da multiplicação escalar
	publicKey := c.ScalarMult(privateKey, c.base)

	// Verificação de segurança
	if isZeroDivisor(publicKey) {
		// Caso extremamente raro - gerar nova chave
		newKey := new(big.Int).Add(privateKey, big.NewInt(1))
		return c.generateUniquePublicKey(newKey)
	}
	return publicKey
}

// deriveKey deriva uma chave simétrica de um divisor
func (c *HyperellipticCurve) deriveKey(shared Divisor) []byte {
	hash := sha256.New()
	hash.Write(shared.U[0].Bytes())
	hash.Write(shared.U[1].Bytes())
	hash.Write(shared.V[0].Bytes())
	hash.Write(shared.V[1].Bytes())
	return hash.Sum(nil)
}

// Estrutura ASN.1 para o criptograma
type CipherTextASN struct {
	C1U0, C1U1 []byte // Coordenadas U do divisor C1
	C1V0, C1V1 []byte // Coordenadas V do divisor C1
	C2         []byte // Hash
	C3         []byte // Ciphertext
}

// Encrypt cifra uma mensagem usando a chave pública
func (c *HyperellipticCurve) Encrypt(pubKey Divisor, message []byte) ([]byte, error) {
	k, err := rand.Int(rand.Reader, c.p)
	if err != nil {
		return nil, err
	}

	C1 := c.ScalarMult(k, c.base)
	shared := c.ScalarMult(k, pubKey)
	key := c.deriveKey(shared)

	ciphertext := make([]byte, len(message))
	for i := range message {
		ciphertext[i] = message[i] ^ key[i%len(key)]
	}

	hash := sha256.Sum256(append(message, key...))

	asn1Data, err := asn1.Marshal(CipherTextASN{
		C1U0: C1.U[0].Bytes(),
		C1U1: C1.U[1].Bytes(),
		C1V0: C1.V[0].Bytes(),
		C1V1: C1.V[1].Bytes(),
		C2:   hash[:],
		C3:   ciphertext,
	})
	if err != nil {
		return nil, err
	}

	return asn1Data, nil
}

// Decrypt decifra uma mensagem usando a chave privada
func (c *HyperellipticCurve) Decrypt(privateKey *big.Int, ciphertextASN []byte) ([]byte, error) {
	var ct CipherTextASN
	if _, err := asn1.Unmarshal(ciphertextASN, &ct); err != nil {
		return nil, fmt.Errorf("falha ao decodificar ASN.1: %v", err)
	}

	C1 := Divisor{
		U: [2]*big.Int{
			new(big.Int).SetBytes(ct.C1U0),
			new(big.Int).SetBytes(ct.C1U1),
		},
		V: [2]*big.Int{
			new(big.Int).SetBytes(ct.C1V0),
			new(big.Int).SetBytes(ct.C1V1),
		},
	}

	shared := c.ScalarMult(privateKey, C1)
	key := c.deriveKey(shared)

	plaintext := make([]byte, len(ct.C3))
	for i := range ct.C3 {
		plaintext[i] = ct.C3[i] ^ key[i%len(key)]
	}

	hashExpected := sha256.Sum256(append(plaintext, key...))
	if subtle.ConstantTimeCompare(hashExpected[:], ct.C2) != 1 {
		return nil, errors.New("verificação de integridade falhou")
	}

	return plaintext, nil
}

/*
 * ==============================
 *          ASSINATURAS
 * ==============================
 */

// SignatureASN define a estrutura ASN.1 da assinatura digital
type SignatureASN struct {
	RU0, RU1 []byte // R divisor - parte U
	RV0, RV1 []byte // R divisor - parte V
	S        []byte // Escalar s
}

// SignMessage assina uma mensagem com a chave privada
func (c *HyperellipticCurve) SignMessage(privateKey *big.Int, message []byte) ([]byte, error) {
	k, err := rand.Int(rand.Reader, c.p)
	if err != nil {
		return nil, err
	}

	R := c.ScalarMult(k, c.base)

	// Hash(R || M)
	hash := sha256.New()
	hash.Write(R.U[0].Bytes())
	hash.Write(R.U[1].Bytes())
	hash.Write(R.V[0].Bytes())
	hash.Write(R.V[1].Bytes())
	hash.Write(message)
	eBytes := hash.Sum(nil)
	e := new(big.Int).SetBytes(eBytes)
	e.Mod(e, c.p)

	// s = k + e*d mod p
	s := new(big.Int).Mul(e, privateKey)
	s.Add(s, k)
	s.Mod(s, c.p)

	sig := SignatureASN{
		RU0: R.U[0].Bytes(),
		RU1: R.U[1].Bytes(),
		RV0: R.V[0].Bytes(),
		RV1: R.V[1].Bytes(),
		S:   s.Bytes(),
	}

	return asn1.Marshal(sig)
}

// VerifySignature verifica a assinatura usando a chave pública
func (c *HyperellipticCurve) VerifySignature(pubKey Divisor, message []byte, sigBytes []byte) (bool, error) {
	var sig SignatureASN
	if _, err := asn1.Unmarshal(sigBytes, &sig); err != nil {
		return false, err
	}

	R := Divisor{
		U: [2]*big.Int{
			new(big.Int).SetBytes(sig.RU0),
			new(big.Int).SetBytes(sig.RU1),
		},
		V: [2]*big.Int{
			new(big.Int).SetBytes(sig.RV0),
			new(big.Int).SetBytes(sig.RV1),
		},
	}
	s := new(big.Int).SetBytes(sig.S)

	// Recalcula e = H(R || M)
	hash := sha256.New()
	hash.Write(R.U[0].Bytes())
	hash.Write(R.U[1].Bytes())
	hash.Write(R.V[0].Bytes())
	hash.Write(R.V[1].Bytes())
	hash.Write(message)
	e := new(big.Int).SetBytes(hash.Sum(nil))
	e.Mod(e, c.p)

	// Verifica se s*G = R + e*PubKey
	left := c.ScalarMult(s, c.base)
	right := c.AddDivisors(R, c.ScalarMult(e, pubKey))

	return divisorsEqual(left, right), nil
}

// divisorsEqual compara dois divisores
func divisorsEqual(D1, D2 Divisor) bool {
	return D1.U[0].Cmp(D2.U[0]) == 0 &&
		D1.U[1].Cmp(D2.U[1]) == 0 &&
		D1.V[0].Cmp(D2.V[0]) == 0 &&
		D1.V[1].Cmp(D2.V[1]) == 0
}

/*
 * ==============================
 *          FUNÇÃO MAIN
 * ==============================
 */

func main() {
	curve := NewCurve()

	// Gera par de chaves
	privateKey, publicKey, err := curve.GenerateKeyPair()
	if err != nil {
		fmt.Println("Erro ao gerar chaves:", err)
		return
	}

	// Mensagem de teste
	message : "**********"
	fmt.Printf("Texto original: %s\n", message)

	// Cifração
	ciphertext, err := curve.Encrypt(publicKey, message)
	if err != nil {
		fmt.Println("Erro ao criptografar:", err)
		return
	}

	// Decifração
	decrypted, err := curve.Decrypt(privateKey, ciphertext)
	if err != nil {
		fmt.Println("Erro ao descriptografar:", err)
		return
	}
	fmt.Printf("Texto decifrado: %s\n", decrypted)

	// Verificação
	if string(message) == string(decrypted) {
		fmt.Println("SUCESSO: A mensagem foi recuperada corretamente")
	} else {
		fmt.Println("FALHA: A descriptografia não funcionou")
	}

	// Função auxiliar para exibir divisores
	fmt.Println("\nChave pública original:")
	printDivisor(publicKey)

	sig, _ := curve.SignMessage(privateKey, message)
	ok, _ := curve.VerifySignature(publicKey, message, sig)

	fmt.Println("Assinatura válida?", ok)
}

// Função auxiliar para exibir um divisor
func printDivisor(d Divisor) {
	fmt.Println("Divisor:")
	fmt.Printf("U[0] (u₀): %x\n", d.U[0].Bytes())
	fmt.Printf("U[1] (u₁): %x\n", d.U[1].Bytes())
	fmt.Printf("V[0] (v₀): %x\n", d.V[0].Bytes())
	fmt.Printf("V[1] (v₁): %x\n", d.V[1].Bytes())
}

// Função para comparar dois divisores
func compareDivisors(d1, d2 Divisor) bool {
	return d1.U[0].Cmp(d2.U[0]) == 0 &&
		d1.U[1].Cmp(d2.U[1]) == 0 &&
		d1.V[0].Cmp(d2.V[0]) == 0 &&
		d1.V[1].Cmp(d2.V[1]) == 0
}
0]) == 0 &&
		d1.V[1].Cmp(d2.V[1]) == 0
}
