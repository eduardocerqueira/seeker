//date: 2023-11-13T16:56:27Z
//url: https://api.github.com/gists/f2ac1c99c63291692b18fdae715338af
//owner: https://api.github.com/users/mickmister

package main

func (p *Plugin) KVScratch() {
	p.API.KVSet("mykey", []byte("my value"))

	initialValue, _ := p.API.KVGet("initial")
	p.API.KVSet("fetched", initialValue)
}
