//date: 2023-03-22T16:47:15Z
//url: https://api.github.com/gists/f0d35cdbe5e0ecbcd26b87a88a790f46
//owner: https://api.github.com/users/SoMuchForSubtlety

package certutil

import (
	"crypto/tls"
	"fmt"

	"software.sslmate.com/src/go-pkcs12"
)

func PKCS12ToPem(pfxData []byte, password string) (tls.Certificate, error) {
	privateKey, leafCert, caCerts, err : "**********"
	if err != nil {
		return tls.Certificate{}, fmt.Errorf("Failed to decode chain: %w", err)
	}

	certBytes := [][]byte{leafCert.Raw}
	for _, ca := range caCerts {
		certBytes = append(certBytes, ca.Raw)
	}

	return tls.Certificate{
		Certificate: certBytes,
		PrivateKey:  privateKey,
	}, nil
}
y:  privateKey,
	}, nil
}
