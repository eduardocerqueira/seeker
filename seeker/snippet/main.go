//date: 2023-01-10T17:07:07Z
//url: https://api.github.com/gists/5b4471c894f171c7cf5fefec5726c98e
//owner: https://api.github.com/users/gen1us2k

package main

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"

	"gopkg.in/yaml.v2"
)

type (
	Cluster struct {
		Cert   string `yaml:"certificate-authority-data"`
		Server string `yaml:"server"`
	}
	ClusterExec struct {
		Name    string  `yaml:"name"`
		Cluster Cluster `yaml:cluster"`
	}
	UserExec struct {
		Name string `yaml:"name"`
		User User   `yaml:"user"`
	}
	User struct {
		Cert string `yaml:"client-certificate-data"`
		Key  string `yaml:"client-key-data"`
	}

	KubeConfig struct {
		Clusters       []ClusterExec `yaml:"clusters"`
		Users          []UserExec    `yaml:"users"`
		CurrentContext string        `yaml:"current-context"`
	}
	Ingress struct {
		Hostname string `json:"hostname"`
		IP       string `json:"ip"`
	}
	Status struct {
		LoadBalancer struct {
			Ingress []Ingress `json:"ingress"`
		} `json:"loadBalancer"`
	}
	StatusResponse struct {
		Status Status `json:"status"`
	}
)

func main() {
	dat, err := os.ReadFile("/Users/gen1us2k/.kube/config")
	if err != nil {
		log.Fatal(err)
	}
	var config KubeConfig
	err = yaml.Unmarshal(dat, &config)
	if err != nil {
		log.Fatal(err)
	}
	var u UserExec
	for _, user := range config.Users {
		if user.Name == config.CurrentContext {
			u = user
		}
	}
	rawCert, err := base64.StdEncoding.DecodeString(u.User.Cert)
	if err != nil {
		log.Fatal(err)
	}
	rawKey, err := base64.StdEncoding.DecodeString(u.User.Key)
	if err != nil {
		log.Fatal(err)
	}

	cert, err := tls.X509KeyPair(rawCert, rawKey)
	if err != nil {
		log.Fatal(err)
	}
	var cluster ClusterExec
	for _, c := range config.Clusters {
		if c.Name == config.CurrentContext {
			cluster = c
		}
	}

	caCertPool := x509.NewCertPool()
	rawCA, err := base64.StdEncoding.DecodeString(cluster.Cluster.Cert)
	if err != nil {
		log.Fatal(err)
	}
	caCertPool.AppendCertsFromPEM(rawCA)
	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		RootCAs:      caCertPool,
	}
	tlsConfig.BuildNameToCertificate()
	transport := &http.Transport{TLSClientConfig: tlsConfig}
	client := &http.Client{Transport: transport}
	resp, err := client.Get(fmt.Sprintf("%s/api/v1/namespaces/default/services/monitoring-service", cluster.Cluster.Server))
	if err != nil {
		log.Fatal(err)
	}
	defer resp.Body.Close()

	// Dump response
	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Fatal(err)
	}
	var status StatusResponse
	json.Unmarshal(data, &status)
	fmt.Println(status.Status.LoadBalancer.Ingress[0].IP)
	fmt.Println(status.Status.LoadBalancer.Ingress[0].Hostname)

}
