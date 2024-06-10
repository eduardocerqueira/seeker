//date: 2024-06-10T17:03:34Z
//url: https://api.github.com/gists/341a70faa0754f7f7a143e834e744bdf
//owner: https://api.github.com/users/cmurphy

package main

import (
        "crypto"
        "crypto/rand"
        "crypto/rsa"
        "crypto/sha256"
        "crypto/x509"
        "crypto/x509/pkix"
        "encoding/asn1"
        "fmt"
        "math/big"

        ct "github.com/google/certificate-transparency-go"
        "github.com/google/certificate-transparency-go/ctutil"
        "github.com/google/certificate-transparency-go/tls"
        "github.com/google/certificate-transparency-go/trillian/ctfe"
        ctx509 "github.com/google/certificate-transparency-go/x509"
        "github.com/google/certificate-transparency-go/x509util"
        ctx509util "github.com/google/certificate-transparency-go/x509util"
        "github.com/sigstore/sigstore/pkg/cryptoutils"
)

func main() {
        privateKey, err := rsa.GenerateKey(rand.Reader, 2048) // generate private key
        if err != nil {
                panic(err)
        }
        skid, err := cryptoutils.SKID(&privateKey.PublicKey)
        if err != nil {
                panic(err)
        }
        preCert := &x509.Certificate{ // pre-certificate template (not technically a real pre-certificate because it doesn't have the poison extension)
                SerialNumber: big.NewInt(1),
                SubjectKeyId: skid,
        }
        preCertDERBytes, err := x509.CreateCertificate(rand.Reader, preCert, preCert, &privateKey.PublicKey, privateKey) // generate the pre-certificate and output the DER encoding
        if err != nil {
                panic(err)
        }
        logID, err := ctfe.GetCTLogID(&privateKey.PublicKey) // get the hash of the public key for the SCT
        if err != nil {
                panic(err)
        }
        sctInput := ct.SignedCertificateTimestamp{ // template for SCT
                SCTVersion: ct.V1,
                Timestamp:  12345,
                LogID:      ct.LogID{KeyID: logID},
        }
        parsedPreCert, err := x509.ParseCertificate(preCertDERBytes) // parse the signed pre-certificate
        if err != nil {
                panic(err)
        }
        logEntry := ct.LogEntry{ // create a LogEntry object containing the TBS certificate
                Leaf: ct.MerkleTreeLeaf{
                        Version:  ct.V1,
                        LeafType: ct.TimestampedEntryLeafType,
                        TimestampedEntry: &ct.TimestampedEntry{
                                Timestamp: 12345,
                                EntryType: ct.PrecertLogEntryType,
                                PrecertEntry: &ct.PreCert{
                                        IssuerKeyHash:  sha256.Sum256(parsedPreCert.RawSubjectPublicKeyInfo),
                                        TBSCertificate: parsedPreCert.RawTBSCertificate,
                                },
                        },
                },
        }
        data, err := ct.SerializeSCTSignatureInput(sctInput, logEntry) // serialize the SCT template and log entry, this is the data to be signed
        if err != nil {
                panic(err)
        }
        h := sha256.Sum256(data)                                            // hash it
        signature, err := privateKey.Sign(rand.Reader, h[:], crypto.SHA256) // sign it
        if err != nil {
                panic(err)
        }
        sct := ct.SignedCertificateTimestamp{ // create the actual SCT with the signature
                SCTVersion: ct.V1,
                LogID:      ct.LogID{KeyID: logID},
                Timestamp:  12345,
                Signature: ct.DigitallySigned{
                        Algorithm: tls.SignatureAndHashAlgorithm{
                                Hash:      tls.SHA256,
                                Signature: tls.RSA,
                        },
                        Signature: signature,
                },
        }
        scts := []*ct.SignedCertificateTimestamp{&sct} // marshal the SCT into an ASN.1 format to put into the x.509 extension
        sctList, err := ctx509util.MarshalSCTsIntoSCTList(scts)
        if err != nil {
                panic(err)
        }
        sctBytes, err := tls.Marshal(*sctList)
        if err != nil {
                panic(err)
        }
        asnSCT, err := asn1.Marshal(sctBytes)
        if err != nil {
                panic(err)
        }
        cert := &x509.Certificate{ // real certificate template with the SCT extension
                SerialNumber: big.NewInt(1),
                SubjectKeyId: skid,
                ExtraExtensions: []pkix.Extension{
                        {
                                Id:    asn1.ObjectIdentifier(ctx509.OIDExtensionCTSCT),
                                Value: asnSCT,
                        },
                },
        }
        certDERBytes, err := x509.CreateCertificate(rand.Reader, cert, cert, &privateKey.PublicKey, privateKey) // generate the real cert, output the DER
        if err != nil {
                panic(err)
        }
        parsedSCTs, err := x509util.ParseSCTsFromCertificate(certDERBytes) // parse the SCT from the certificate
        if err != nil {
                panic(err)
        }
        parsedCert, err := ctx509.ParseCertificate(certDERBytes) // parse the certificate
        if err != nil {
                panic(err)
        }
        parentCert := parsedCert
        chain := []*ctx509.Certificate{parsedCert, parentCert}                    // chain of self-signed certs
        err = ctutil.VerifySCT(&privateKey.PublicKey, chain, parsedSCTs[0], true) // verify the SCT
        if err == nil {
                fmt.Println("verified!")
                return
        }
        fmt.Printf("SCT could not be verified: %v\n", err)
}