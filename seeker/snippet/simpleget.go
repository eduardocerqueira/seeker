//date: 2022-08-31T17:07:46Z
//url: https://api.github.com/gists/ec735d90b0bdb8deeff9a81424e6efef
//owner: https://api.github.com/users/scottopell

package main

import (
        "crypto/tls"
        "fmt"
)

// go build && strace -o strace-out.txt -e trace=file ./simpleget
// usually I get `strace-out.txt`, but sometimes I get `strace-uncommon-out.txt`. Why?
func main() {
        conn, err := tls.Dial("tcp", "www.google.com:443", nil)
        if err != nil {
                fmt.Println("Error in Dial", err)
                return
        }
        defer conn.Close()

        state := conn.ConnectionState()
        fmt.Printf("Connection has %d Verified Chains\n", len(state.VerifiedChains))
        for i, chain := range state.VerifiedChains {
                fmt.Printf("Chain %d:\n", i)
                for _, cert := range chain {
                        fmt.Printf("\tIssuer Name: %s\n", cert.Issuer)
                        fmt.Printf("\tExpiry: %s \n", cert.NotAfter.Format("2006-January-02"))
                        fmt.Printf("\tCommon Name: %s \n", cert.Issuer.CommonName)
                        fmt.Println("\t-------------")
                }
        }
}