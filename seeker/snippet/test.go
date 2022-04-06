//date: 2022-04-06T17:13:59Z
//url: https://api.github.com/gists/9fd9ed8c01802fb7d36d98ed268f3a18
//owner: https://api.github.com/users/jeffreytolar

package main

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"log"
	"net"
	"time"

	"github.com/cbeuw/connutil"
	"golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/agent"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func newSigner() (*ecdsa.PrivateKey, ssh.Signer) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	check(err)

	signer, err := ssh.NewSignerFromKey(key)
	check(err)

	return key, signer
}

func makeCert() *ssh.Certificate {
	cert := &ssh.Certificate{
		Key:             userSigner.PublicKey(),
		Serial:          1,
		CertType:        ssh.UserCert,
		KeyId:           "testkey",
		ValidPrincipals: []string{"test"},
		ValidAfter:      uint64(time.Now().Add(-time.Hour).Unix()),
		ValidBefore:     uint64(time.Now().Add(time.Hour).Unix()),
	}
	err := cert.SignCert(rand.Reader, userCASigner)
	check(err)
	return cert
}

var (
	userCAKey, userCASigner = newSigner()
	hostKey, hostSigner     = newSigner()
	userKey, userSigner     = newSigner()

	userCert = makeCert()
)

func server(nConn net.Conn) {
	config := &ssh.ServerConfig{
		PublicKeyCallback: (&ssh.CertChecker{
			IsUserAuthority: func(auth ssh.PublicKey) bool {
				return bytes.Equal(auth.Marshal(), userCASigner.PublicKey().Marshal())
			},
		}).Authenticate,
	}

	config.AddHostKey(hostSigner)

	_, _, _, err := ssh.NewServerConn(nConn, config)
	check(err)
	log.Print("[server] Completed handshake")
}

func client(nConn net.Conn, signers []ssh.Signer) {
	config := &ssh.ClientConfig{
		User:            "test",
		HostKeyCallback: ssh.FixedHostKey(hostSigner.PublicKey()),
		Auth:            []ssh.AuthMethod{ssh.PublicKeys(signers...)},
	}

	_, _, _, err := ssh.NewClientConn(nConn, "pipe", config)
	check(err)
	log.Print("[client] Completed handshake")
}

func signersLocalKey() []ssh.Signer {
	certSigner, err := ssh.NewCertSigner(userCert, userSigner)
	check(err)
	return []ssh.Signer{certSigner}
}

func signersKeyring() []ssh.Signer {
	ag := agent.NewKeyring()
	check(ag.Add(agent.AddedKey{
		PrivateKey:  userKey,
		Certificate: userCert,
	}))
	signers, err := ag.Signers()
	check(err)
	return signers
}

func signersRemoteAgent() []ssh.Signer {
	ag := agent.NewKeyring()
	check(ag.Add(agent.AddedKey{
		Certificate: userCert,
		PrivateKey:  userKey,
	}))

	client, server := connutil.AsyncPipe()

	go func() {
		check(agent.ServeAgent(ag, server))
	}()

	signers, err := agent.NewClient(client).Signers()
	check(err)
	return signers
}

func main() {
	clientConn, serverConn := connutil.AsyncPipe()
	go server(serverConn)
	// client(clientConn, signersLocalKey()) // this works
	// client(clientConn, signersKeyring()) // this works too
	client(clientConn, signersRemoteAgent()) // this fails: ssh: handshake failed: agent: unsupported algorithm "ecdsa-sha2-nistp256"
}