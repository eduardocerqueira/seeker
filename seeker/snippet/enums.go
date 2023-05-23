//date: 2023-05-23T16:47:38Z
//url: https://api.github.com/gists/ca8ceeab2d684a748f747d1244cbf5ba
//owner: https://api.github.com/users/acuscorp



package main

import (
	"fmt"
)

type State int
type App struct {
	Name  string
	State State
}

const (
	OnStart State = iota
	OnRun
	OnStop
	OnFinish
)

func (s State) String() string {
	switch s {
	case OnStart:
		return "OnStart"
	case OnRun:
		return "OnRun"
	case OnStop:
		return "OnStop"
	case OnFinish:
		return "OnFinish"
	}
	return "Unknown"
}

func (s State) PrintState() {
	fmt.Println("App state is " + s.String())
}

func (a App) OnStart() {
	a.State = OnStart
	a.State.PrintState()

}

func (a App) OnRun() {
	a.State = OnRun
	a.State.PrintState()
}

func (a App) OnStop() {
	a.State = OnStop
	a.State.PrintState()
}

func (a App) OnFinish() {
	a.State = OnFinish
	a.State.PrintState()
}

func main() {
	app := App{Name: "TestingEnums"}
	app.OnStart()
	app.OnRun()
	app.OnStop()
	app.OnFinish()
}
