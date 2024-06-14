//date: 2024-06-14T17:02:04Z
//url: https://api.github.com/gists/3d27f29283588cf1f162e2b0e985a841
//owner: https://api.github.com/users/ChicK00o

package additionalFields

import (
	"encoding/json"
	"errors"
	"reflect"
)

type AdditionalFields struct {
	Fields map[string]json.RawMessage `json:"-"`
}

// UnmarshalJSONWithAdditionalFields unmarshals known and additional fields
func UnmarshalJSONWithAdditionalFields(data []byte, target interface{}) error {
	// Create a map to hold the data
	var allFields map[string]json.RawMessage
	if err := json.Unmarshal(data, &allFields); err != nil {
		return err
	}

	// Use reflection to handle known fields
	targetType := reflect.TypeOf(target).Elem()
	targetValue := reflect.ValueOf(target).Elem()

	// Iterate through the fields of the target struct
	for i := 0; i < targetType.NumField(); i++ {
		field := targetType.Field(i)
		jsonTag := field.Tag.Get("json")

		if jsonTag != "" && jsonTag != "-" {
			if value, found := allFields[jsonTag]; found {
				// Unmarshal known field value
				fieldValue := targetValue.Field(i).Addr().Interface()
				if err := json.Unmarshal(value, fieldValue); err != nil {
					return err
				}
				// Remove from allFields map after processing
				delete(allFields, jsonTag)
			}
		}
	}

	// Set additional fields
	if afField := targetValue.FieldByName("Fields"); afField.IsValid() && afField.CanSet() {
		afField.Set(reflect.ValueOf(allFields))
	}

	return nil
}

// MarshalJSONWithAdditionalFieldsDynamic marshals known and additional fields dynamically
func MarshalJSONWithAdditionalFieldsDynamic(source interface{}) ([]byte, error) {
	// Ensure source is a pointer
	sourceValue := reflect.ValueOf(source)
	if sourceValue.Kind() != reflect.Ptr {
		return nil, errors.New("source must be a pointer")
	}
	sourceValue = sourceValue.Elem()

	// Extract additional fields
	additionalFields, ok := sourceValue.FieldByName("AdditionalFields").Interface().(AdditionalFields)
	if !ok {
		return nil, errors.New("invalid type for additional fields")
	}

	// Create a new struct type dynamically for the alias, excluding the "AdditionalFields" field
	sourceType := sourceValue.Type()
	fields := make([]reflect.StructField, 0, sourceType.NumField())
	for i := 0; i < sourceType.NumField(); i++ {
		field := sourceType.Field(i)
		if field.Name != "AdditionalFields" {
			fields = append(fields, field)
		}
	}
	aliasType := reflect.StructOf(fields)

	// Create a new instance of the alias type
	aliasInstance := reflect.New(aliasType).Elem()

	// Copy the known fields to the alias instance
	for i := 0; i < sourceType.NumField(); i++ {
		field := sourceType.Field(i)
		if field.Name != "AdditionalFields" {
			aliasInstance.FieldByName(field.Name).Set(sourceValue.Field(i))
		}
	}

	// Marshal known fields first
	knownFields, err := json.Marshal(aliasInstance.Addr().Interface())
	if err != nil {
		return nil, err
	}

	// Unmarshal known fields into a map
	var knownMap map[string]json.RawMessage
	if err := json.Unmarshal(knownFields, &knownMap); err != nil {
		return nil, err
	}

	// Merge known and additional fields
	for key, value := range additionalFields.Fields {
		knownMap[key] = value
	}

	// Marshal the combined map
	return json.Marshal(knownMap)
}
