//date: 2025-05-12T16:38:30Z
//url: https://api.github.com/gists/fcda33d60207eef9b0c05f51aa6a32ea
//owner: https://api.github.com/users/pedroalbanese

package main

import (
	"crypto/rand"
	"crypto/sha256"
	"fmt"

	"github.com/cloudflare/circl/ecc/bls12381"
	"github.com/cloudflare/circl/ecc/bls12381/ff"
)

// Gera um escalar aleatório seguro
func randomScalar() *ff.Scalar {
	scalar := new(ff.Scalar)
	buf := make([]byte, 32) // BLS12-381 usa 32 bytes
	_, err := rand.Read(buf)
	if err != nil {
		panic("Erro ao gerar número aleatório")
	}
	scalar.SetBytes(buf)
	return scalar
}

// Hash para Scalar (usado para derivar valores do ID)
func hashToScalar(ID string) *ff.Scalar {
	hash := sha256.Sum256([]byte(ID))
	scalar := new(ff.Scalar)
	scalar.SetBytes(hash[:])
	return scalar
}

// Gera chave privada de um usuário a partir da chave mestra
func generatePrivateKey(masterKey *ff.Scalar, userID string) *ff.Scalar {
	userScalar := hashToScalar(userID)
	privateKey := new(ff.Scalar)
	privateKey.Mul(masterKey, userScalar) // sk = msk ⋅ H(ID)
	return privateKey
}

// Gera chave pública de um usuário a partir da chave mestra
func generatePublicKeyForUser(masterPublicKey *bls12381.G2, userID string) *bls12381.G2 {
	userScalar := hashToScalar(userID)
	publicKey := new(bls12381.G2)
	publicKey.ScalarMult(userScalar, masterPublicKey) // pk = H(ID) ⋅ mpk
	return publicKey
}

// Gera fator de cegamento aleatório
func generateBlindFactor() *ff.Scalar {
	return randomScalar()
}

// Converte mensagem para um ponto na curva G₁
func hashToG1(message []byte) *bls12381.G1 {
	hashMessage := new(bls12381.G1)
	hashMessage.Hash(message, nil)
	return hashMessage
}

// Cega a mensagem
func blindMessage(originalG1 *bls12381.G1, blindFactor *ff.Scalar) *bls12381.G1 {
	blindedMessage := new(bls12381.G1)
	blindedMessage.ScalarMult(blindFactor, originalG1) // m' = r ⋅ H(m)
	return blindedMessage
}

// Assinar mensagem cegada com a chave privada
func signBlindedMessage(blindedMessage *bls12381.G1, privKey *ff.Scalar) *bls12381.G1 {
	blindedSignature := new(bls12381.G1)
	blindedSignature.ScalarMult(privKey, blindedMessage) // σ' = sk ⋅ m' = sk ⋅ r ⋅ H(m)
	return blindedSignature
}

// Descega a assinatura cegada
func unblindSignature(blindedSignature *bls12381.G1, blindFactor *ff.Scalar) *bls12381.G1 {
	inverseBlindFactor := new(ff.Scalar)
	inverseBlindFactor.Inv(blindFactor) // r⁻¹

	finalSignature := new(bls12381.G1)
	finalSignature.ScalarMult(inverseBlindFactor, blindedSignature) // σ = r⁻¹ ⋅ σ' = sk ⋅ H(m)
	return finalSignature
}

// Verifica assinatura
func verifySignature(message []byte, signature *bls12381.G1, publicKey *bls12381.G2) bool {
	hashMessage := hashToG1(message)

	e1 := bls12381.Pair(signature, bls12381.G2Generator()) // e(σ, G₂)
	e2 := bls12381.Pair(hashMessage, publicKey)            // e(H(m), pk)

	return e1.IsEqual(e2)
}

// Agrega assinaturas
func aggregateSignatures(signatures []*bls12381.G1) *bls12381.G1 {
	aggSig := new(bls12381.G1)
	aggSig.SetIdentity()
	for _, sig := range signatures {
		aggSig.Add(aggSig, sig)
	}
	return aggSig
}

// Verifica assinatura agregada
func verifyAggregateSignature(messages [][]byte, aggSignature *bls12381.G1, pubKeys []*bls12381.G2) bool {
	if len(messages) != len(pubKeys) {
		return false
	}

	// Calcular o produto dos emparelhamentos e(H(m_i), pk_i)
	prodPairing := bls12381.Pair(hashToG1(messages[0]), pubKeys[0])
	for i := 1; i < len(messages); i++ {
		temp := bls12381.Pair(hashToG1(messages[i]), pubKeys[i])
		prodPairing.Mul(prodPairing, temp)
	}

	// Calcular e(σ_agg, G2)
	sigPairing := bls12381.Pair(aggSignature, bls12381.G2Generator())

	return sigPairing.IsEqual(prodPairing)
}

func main() {
	// Configuração da chave mestra
	masterKey := new(ff.Scalar)
	masterKey.SetUint64(1234567890)
	basePointG2 := bls12381.G2Generator()
	var masterPublicKey bls12381.G2
	masterPublicKey.ScalarMult(masterKey, basePointG2)

	// Criar chaves individuais
	userIDs := []string{"alice@example.com", "bob@example.com", "carol@example.com"}
	var pubKeys []*bls12381.G2
	var privKeys []*ff.Scalar

	for _, ID := range userIDs {
		privKey := generatePrivateKey(masterKey, ID)
		pubKey := generatePublicKeyForUser(&masterPublicKey, ID)

		privKeys = append(privKeys, privKey)
		pubKeys = append(pubKeys, pubKey)
	}

	// Criar mensagens
	messages := [][]byte{[]byte("Voto: Alice, Opção A"), []byte("Voto: Bob, Opção B"), []byte("Voto: Carol, Opção A")}
	var originalHashes []*bls12381.G1
	var blindFactors []*ff.Scalar
	var blindedMessages []*bls12381.G1
	var blindedSignatures []*bls12381.G1
	var finalSignatures []*bls12381.G1

	// Processo de cegamento e assinatura
	for i, msg := range messages {
		// 1. Hash da mensagem original
		originalHash := hashToG1(msg)
		originalHashes = append(originalHashes, originalHash)

		// 2. Gerar fator de cegamento
		blindFactor := generateBlindFactor()
		blindFactors = append(blindFactors, blindFactor)

		// 3. Cegar a mensagem
		blindedMsg := blindMessage(originalHash, blindFactor)
		blindedMessages = append(blindedMessages, blindedMsg)

		// 4. Assinar a mensagem cegada (feito pela autoridade)
		blindedSig := signBlindedMessage(blindedMsg, privKeys[i])
		blindedSignatures = append(blindedSignatures, blindedSig)

		// 5. Descegar a assinatura (feito pelo usuário)
		finalSig := unblindSignature(blindedSig, blindFactor)
		finalSignatures = append(finalSignatures, finalSig)
	}

	// Verificação das assinaturas individuais
	fmt.Println("Verificação de assinaturas individuais:")
	for i, msg := range messages {
		if verifySignature(msg, finalSignatures[i], pubKeys[i]) {
			fmt.Printf("  Assinatura válida para %s\n", userIDs[i])
		} else {
			fmt.Printf("  Assinatura inválida para %s\n", userIDs[i])
		}
	}

	// Agregação e verificação de assinaturas
	fmt.Println("\nVerificação de assinatura agregada:")
	aggSignature := aggregateSignatures(finalSignatures)
	if verifyAggregateSignature(messages, aggSignature, pubKeys) {
		fmt.Println("  Assinatura agregada válida!")
	} else {
		fmt.Println("  Assinatura agregada inválida!")
	}

	// Verificação da integridade do processo de cegamento
	fmt.Println("\nVerificação do processo de cegamento:")
	for i := range messages {
		// Verificar se σ' = sk ⋅ r ⋅ H(m)
		expectedBlindedSig := new(bls12381.G1)
		expectedBlindedSig.ScalarMult(privKeys[i], blindedMessages[i])

		if expectedBlindedSig.IsEqual(blindedSignatures[i]) {
			fmt.Printf("  Assinatura cegada correta para %s\n", userIDs[i])
		} else {
			fmt.Printf("  ERRO na assinatura cegada para %s\n", userIDs[i])
		}

		// Verificar se σ = r⁻¹ ⋅ σ' = sk ⋅ H(m)
		expectedFinalSig := new(bls12381.G1)
		expectedFinalSig.ScalarMult(privKeys[i], originalHashes[i])

		if expectedFinalSig.IsEqual(finalSignatures[i]) {
			fmt.Printf("  Assinatura final correta para %s\n", userIDs[i])
		} else {
			fmt.Printf("  ERRO na assinatura final para %s\n", userIDs[i])
		}
	}
}
