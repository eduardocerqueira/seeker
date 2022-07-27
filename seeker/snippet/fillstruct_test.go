//date: 2022-07-27T17:17:33Z
//url: https://api.github.com/gists/2f36281c4454fab3579fbc22d6cafe77
//owner: https://api.github.com/users/bentranter

package fillstruct

import (
	"reflect"
	"testing"
)

// FillStruct fills a struct with a string value as an example of how to use
// reflection to accomplish this.
func FillStruct(v interface{}) {
	value := reflect.Indirect(reflect.ValueOf(v))
	if value.Kind() != reflect.Struct {
		return
	}

	// If the field can be set and is a string, set it to a constant string.
	fieldValue := value.Field(0)
	if fieldValue.CanSet() {
		if fieldValue.Kind() == reflect.String {
			fieldValue.SetString("test")
		}
		return
	}

	// Otherwise, call itself on the struct field.
	var fieldValueInterface interface{}
	switch fieldValue.Kind() {
	case reflect.Struct:
		if fieldValue.CanAddr() {
			fieldValueInterface = fieldValue.Addr().Interface()
		}
	case reflect.Ptr:
		if fieldValue.IsNil() {
			if fieldValue.CanSet() {
				fieldValue.Set(reflect.New(fieldValue.Type().Elem()))
			}
		}
		fieldValueInterface = fieldValue.Interface()
	}

	FillStruct(fieldValueInterface)
}

type S struct {
	A string `edi:"01"`
}

type V struct {
	S S `edi:"S"`
}

type Vstar struct {
	S *S `edi:"S"`
}

func TestFillStruct(t *testing.T) {
	cases := []*V{
		&V{},
		&V{S: S{A: "set"}},
	}

	for _, v := range cases {
		FillStruct(v)
		if v.S.A != "test" {
			t.Fatalf("expected test, got %#v\n", v.S)
		}
	}

	cases2 := []*Vstar{
		&Vstar{},
		&Vstar{S: &S{A: "set"}},
	}

	for _, v := range cases2 {
		FillStruct(v)
		if v.S.A != "test" {
			t.Fatalf("expected test, got %#v\n", v.S)
		}
	}
}

func BenchmarkFillStruct(b *testing.B) {
	v := &V{}

	for i := 0; i < b.N; i++ {
		FillStruct(v)
	}
}
