//date: 2023-09-08T17:02:37Z
//url: https://api.github.com/gists/21e1785b56961bbcbd32b1b7401b53cc
//owner: https://api.github.com/users/jaredwarren-cb

package main

/*
Below is a Go reference implementation that you can input your Coinbase Cloud API key name and private key into. 
*/

import (
	"crypto/rand"
	"crypto/x509"
	"encoding/pem"
	"fmt"
      log "github.com/sirupsen/logrus"	
      "gopkg.in/square/go-jose.v2"
	"gopkg.in/square/go-jose.v2/jwt"
	"math"
	"math/big"
	"time"
)

const (
	nameEnvVar       = "<Replace this with your api key name>"
	privateKeyEnvVar =  "<Replace this with your private key>"
)

type APIKeyClaims struct {
	*jwt.Claims
	URI string `json:"uri"`
}

func  buildJWT(service, uri string) (string, error) {
	block, _ := pem.Decode([]byte(privateKeyEnvVar))
	if block == nil {
		return "", fmt.Errorf("jwt: Could not decode private key")
	}

	key, err := x509.ParseECPrivateKey(block.Bytes)
	if err != nil {
		return "", fmt.Errorf("jwt: %w", err)
	}

	sig, err := jose.NewSigner(
		jose.SigningKey{Algorithm: jose.ES256, Key: key},
		(&jose.SignerOptions{NonceSource: nonceSource{}}).WithType("JWT").WithHeader("kid", nameEnvVar),
	)
	if err != nil {
		return "", fmt.Errorf("jwt: %w", err)
	}

	cl := &APIKeyClaims{
		Claims: &jwt.Claims{
			Subject:   nameEnvVar,
			Issuer:    "coinbase-cloud",
			NotBefore: jwt.NewNumericDate(time.Now()),
			Expiry:    jwt.NewNumericDate(time.Now().Add(1 * time.Minute)),
			Audience:  jwt.Audience{service},
		},
		URI: uri,
	}
	jwtString, err := jwt.Signed(sig).Claims(cl).CompactSerialize()
	if err != nil {
		return "", fmt.Errorf("jwt: %w", err)
	}
	return jwtString, nil
}

var max = big.NewInt(math.MaxInt64)

type nonceSource struct{}

func (n nonceSource) Nonce() (string, error) {
	r, err := rand.Int(rand.Reader, max)
	if err != nil {
		return "", err
	}
	return r.String(), nil
}

func main() {
	uri := fmt.Sprintf("%s %s%s", "POST", "api.developer.coinbase.com", "/api/v3/coinbase.user_activity_report_service.UserActivitiesReportPublicService/ReportUserActivities")

	jwt,err := buildJWT("user_activity_report_service", uri)
      if err != nil {
		log.Errorf("error building jwt: %v", err)
	}
	log.Infof("jwt result: %s", jwt)

}
