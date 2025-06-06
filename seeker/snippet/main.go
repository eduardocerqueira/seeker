//date: 2025-06-06T16:54:37Z
//url: https://api.github.com/gists/1098810b07f2915ca1e95a94afc1374b
//owner: https://api.github.com/users/WinPooh32

// Author: WinPooh32
// Licence: MIT
// Link: https://github.com/WinPooh32
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"text/template"
)

const tpl = `{
	"key_1": {{ float .value1 }},
	"key_2": {{ int .value2 }},
	"key_3": {{ string .value3 }},
	"key_4": {{ object .value4 }},
	"key_5": {{ bool .value5 }},
	"key_6": "My text {{ substr .value6 }}"
}`

func main() {
	t := template.New("sometempl").Option("missingkey=error")
	t = t.Funcs(template.FuncMap{
		"float": func(v float64) (string, error) {
			b, err := json.Marshal(v)
			return string(b), err
		},
		"int": func(v int) (string, error) {
			b, err := json.Marshal(v)
			return string(b), err
		},
		"bool": func(v bool) (string, error) {
			b, err := json.Marshal(v)
			return string(b), err
		},
		"string": func(v string) (string, error) {
			b, err := json.Marshal(v)
			return string(b), err
		},
		"substr": func(v string) (string, error) {
			b, err := json.Marshal(v)
			if err != nil {
				return "", err
			}
			return string(b[1 : len(b)-1]), err
		},
		"object": func(v any) (string, error) {
			if kind := reflect.TypeOf(v).Kind(); kind != reflect.Struct {
				return "", fmt.Errorf("expected struct, but got %s", kind)
			}
			b, err := json.Marshal(v)
			return string(b), err
		},
	})

	t, err := t.Parse(tpl)
	if err != nil {
		panic(err)
	}

	buf := bytes.NewBuffer(nil)

	err = t.Execute(buf, map[string]any{
		"value1": 123.0,
		"value2": 666,
		"value3": "string",
		"value4": struct {
			A string `json:"a,omitempty"`
			B int    `json:"b,omitempty"`
		}{
			A: "some string",
			B: 999,
		},
		"value5": true,
		"value6": `Invalid utf8` + string([]byte{0xC0}) + `		
 text`,
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(buf.String(), "\nis json valid: ", json.Valid(buf.Bytes()))
}
