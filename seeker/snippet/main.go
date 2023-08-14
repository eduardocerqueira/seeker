//date: 2023-08-14T17:08:46Z
//url: https://api.github.com/gists/8495bc8dfa3def64b8493f9cfd16c0e7
//owner: https://api.github.com/users/danyalprout

package main

import (
	"fmt"
	"github.com/naoina/toml"
	"os"
	"time"
)

type tomlConfig struct {
	Foo struct {
		Name string
		Dur  time.Duration
	}
}

func main() {
	f, err := os.Open("/Users/danyal/code/cb/base-observability/rpc-collector/cmd/test/test.toml")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	var config tomlConfig
	if err := toml.NewDecoder(f).Decode(&config); err != nil {
		panic(err)
	}
	fmt.Println("seconds", config.Foo.Dur.Seconds())
	fmt.Println("nano", config.Foo.Dur.Nanoseconds())
}
