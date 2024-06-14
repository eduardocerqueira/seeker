//date: 2024-06-14T17:02:04Z
//url: https://api.github.com/gists/3d27f29283588cf1f162e2b0e985a841
//owner: https://api.github.com/users/ChicK00o

package additionalFields

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

// Person is a sample struct to demonstrate the concept
type TestPerson struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
	AdditionalFields
}

// MarshalJSON custom marshaller for Person
func (p TestPerson) MarshalJSON() ([]byte, error) {
	return MarshalJSONWithAdditionalFieldsDynamic(&p)
}

// UnmarshalJSON custom unmarshaller for Person
func (p *TestPerson) UnmarshalJSON(data []byte) error {
	return UnmarshalJSONWithAdditionalFields(data, p)
}

// Organisation is a sample struct to demonstrate the concept
type TestOrganization struct {
	Name    string       `json:"name"`
	Address string       `json:"address"`
	Persons []TestPerson `json:"persons"`
	AdditionalFields
}

// MarshalJSON custom marshaller for Organisation
func (o TestOrganization) MarshalJSON() ([]byte, error) {
	return MarshalJSONWithAdditionalFieldsDynamic(&o)
}

// UnmarshalJSON custom unmarshaller for Organisation
func (o *TestOrganization) UnmarshalJSON(data []byte) error {
	return UnmarshalJSONWithAdditionalFields(data, o)
}

func TestUnMarshallingAndMarshallingWithAdditionalFields(t *testing.T) {
	// Test JSON for Organisation
	jsonStr := `
	{
		"name": "TechCorp",
		"address": "123 Tech Street",
		"persons": [
			{"name": "John Doe", "age": 30, "role": "Developer"},
			{"name": "Jane Smith", "age": 25, "role": "Designer"}
		],
		"founded": "2001",
		"industry": "Technology"
	}`
	jsonByte := []byte(jsonStr)

	var org TestOrganization
	if err := json.Unmarshal(jsonByte, &org); err != nil {
		fmt.Println("Unmarshal error:", err)
		return
	}

	// Modify the struct (optional)
	org.Name = "TechCorp International"
	org.Persons[0].Age = 31
	org.Persons[0].AdditionalFields.Fields["role"] = json.RawMessage(`"Lead Developer"`)
	org.Persons[1].AdditionalFields.Fields["role"] = json.RawMessage(`"Senior Designer"`)
	org.AdditionalFields.Fields["industry"] = json.RawMessage(`"Tech and Innovation"`)

	marshaledJSON, err := json.Marshal(org)
	if err != nil {
		fmt.Println("Marshal error:", err)
		return
	}

	finalStr := `{"address":"123 Tech Street","founded":"2001","industry":"Tech and Innovation","name":"TechCorp International","persons":[{"age":31,"name":"John Doe","role":"Lead Developer"},{"age":25,"name":"Jane Smith","role":"Senior Designer"}]}`

	assert.JSONEq(t, finalStr, string(marshaledJSON))
}
