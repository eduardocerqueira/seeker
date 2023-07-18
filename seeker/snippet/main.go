//date: 2023-07-18T17:03:51Z
//url: https://api.github.com/gists/68a147f28d75ebc49345a497a130bc91
//owner: https://api.github.com/users/hlubek

package main

import (
	"encoding/json"
	"fmt"
	"reflect"
)

type Something interface{}

type Something1 struct {
	Aaa, Bbb string
}

type Something2 struct {
	Ccc, Ddd string
}

var _ Something = Something1{}
var _ Something = Something2{}

// We need to register all known message types here to be able to unmarshal them to the correct interface type.
var knownImplementations = []Something{
	//	Something1{},
	Something2{},
}

type Container struct {
	Value Something `json:"value"`
}

func (c *Container) UnmarshalJSON(bytes []byte) error {
	var data struct {
		Type  string
		Value json.RawMessage
	}
	if err := json.Unmarshal(bytes, &data); err != nil {
		return err
	}

	for _, knownImplementation := range knownImplementations {
		knownType := reflect.TypeOf(knownImplementation)
		if knownType.String() == data.Type {
			// Create a new pointer to a value of the concrete message type
			target := reflect.New(knownType)
			// Unmarshal the data to an interface to the concrete value (which will act as a pointer, don't ask why)
			if err := json.Unmarshal(data.Value, target.Interface()); err != nil {
				return err
			}
			// Now we get the element value of the target and convert it to the interface type (this is to get rid of a pointer type instead of a plain struct value)
			c.Value = target.Elem().Interface().(Something)
			return nil
		}
	}

	return fmt.Errorf("value type not known: %s", data.Type)
}

func (c Container) MarshalJSON() ([]byte, error) {
	// Marshal to type and actual data to handle unmarshaling to specific interface type
	return json.Marshal(struct {
		Type  string
		Value any
	}{
		Type:  reflect.TypeOf(c.Value).String(),
		Value: c.Value,
	})
}

func main() {
	c := Container{
		Value: Something1{
			Aaa: "aaa",
			Bbb: "bbb",
		},
	}

	data, err := json.Marshal(c)
	if err != nil {
		panic(err)
	}

	var unmarshaled Container
	err = json.Unmarshal(data, &unmarshaled)
	if err != nil {
		panic(err)
	}

	switch v := unmarshaled.Value.(type) {
	case Something1:
		println(v.Aaa)
	default:
		panic(fmt.Sprintf("unexpected value type: %T", v))
	}
}
