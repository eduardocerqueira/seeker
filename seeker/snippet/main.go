//date: 2024-06-21T17:03:49Z
//url: https://api.github.com/gists/9e5c6c95d027580220f5461e98a15ae9
//owner: https://api.github.com/users/nklaassen

package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"log"
	"math/big"
	"net/http"
	"time"
)

func main() {
	cert, err := generateCert()
	if err != nil {
		log.Fatal(err)
	}
	listener, err := tls.Listen("tcp", "localhost:0", &tls.Config{
		Certificates: []tls.Certificate{*cert},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Listening on https://%s\n", listener.Addr().String())

	srv := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			if _, err := w.Write([]byte("<h1>Hello, World!")); err != nil {
				log.Println(err)
			}
		}),
	}
	log.Fatal(srv.Serve(listener))
}

func generateCert() (*tls.Certificate, error) {
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("generating key: %w", err)
	}
	template := &x509.Certificate{
		DNSNames:     []string{"localhost"},
		NotAfter:     time.Now().Add(365 * 24 * time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		SerialNumber: big.NewInt(1),
	}
	x509CertBytes, err := x509.CreateCertificate(rand.Reader, template, template, priv.Public(), priv)
	if err != nil {
		return nil, fmt.Errorf("creating cert: %w", err)
	}
	cert := &tls.Certificate{
		Certificate: [][]byte{x509CertBytes},
		PrivateKey:  priv,
	}
	return cert, nil
}
