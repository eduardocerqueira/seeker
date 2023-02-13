//date: 2023-02-13T17:06:11Z
//url: https://api.github.com/gists/bdc3a6391682d351516ebcf766229e5b
//owner: https://api.github.com/users/Integralist

package main

import (
	"time"

	"github.com/theckman/yacspin"
)

func main() {
	spinner, _ := yacspin.New(yacspin.Config{
		CharSet:           yacspin.CharSets[9],
		Frequency:         100 * time.Millisecond,
		StopCharacter:     "✓",
		StopColors:        []string{"fgGreen"},
		StopFailCharacter: "✗",
		StopFailColors:    []string{"fgRed"},
		Suffix:            " ",
		// NotTTY:            true,
	})

	_ = spinner.Start()
	spinner.Message("1.")

	time.Sleep(4 * time.Second)

	spinner.Message("2.")

	time.Sleep(4 * time.Second)

	spinner.StopMessage("2.")

	_ = spinner.Stop()

	_ = spinner.Start()

	spinner.Message("3.")

	time.Sleep(4 * time.Second)

	spinner.StopFailMessage("3.")

	_ = spinner.StopFail()
}