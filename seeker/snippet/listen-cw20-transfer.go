//date: 2021-09-02T16:54:53Z
//url: https://api.github.com/gists/44ec1848124ca0b531792c679e38591b
//owner: https://api.github.com/users/giansalex

package main

import (
	"context"
	"fmt"
	"log"
	"time"

	rpchttp "github.com/tendermint/tendermint/rpc/client/http"
)

func main() {
	client, err := rpchttp.New("tcp://127.0.0.1:26657", "/websocket")
	cw20Contract := "juno17pyrfhc3x94mzj2lzqhj89hdkpjxffede7y5y8"
	if err != nil {
		log.Fatal(err)
	}

	err = client.Start()
	if err != nil {
		log.Fatal(err)
	}
	defer client.Stop()
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	query := fmt.Sprintf("message.module='wasm' AND message.contract_address='%s' AND wasm.action='transfer'", cw20Contract)
	txs, err := client.Subscribe(ctx, "test-client", query)
	if err != nil {
		log.Fatal(err)
	}

	for e := range txs {
		from := e.Events["wasm.from"][0]
		to := e.Events["wasm.to"][0]
		amount := e.Events["wasm.amount"][0]

		fmt.Println("Tx: " + e.Events["tx.hash"][0])
		fmt.Printf("Sender: %s \nRecipient: %s \nAmount: %s \n", from, to, amount)
	}
}
