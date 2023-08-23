//date: 2023-08-23T17:05:24Z
//url: https://api.github.com/gists/4c94db8053aa8f6e8cdc5e6b5b3f4c8c
//owner: https://api.github.com/users/AcousticTypewriter

package main

import (
        "machine"
        "time" 
 )

func door () {
  //run door motor
}

func elevatorYO () {
  //run elevator motor
}


func main() {
  cond := true
  for {
    result, err := machine.GetRNG()
    if err != nil {
      println("uh oh")
    } else if result == 10 {
      cond = false
    }

    switch cond {
      case true:
        go door()
      case false:
        go elevatorYO()
    }
    time.Sleep(time.Millisecond*1)
  }  
}
