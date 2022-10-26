//date: 2022-10-26T17:07:57Z
//url: https://api.github.com/gists/5f3ec81b6196f8c1c17f86342ed43d34
//owner: https://api.github.com/users/nicolasparada

package property

import (
	"fmt"
	"strings"

	"gopkg.in/yaml.v3"
)

// Properties list.
type Properties []Property

// MarshalYAML implements yaml.Marshaler interface
// to marshall a sorted list of properties into an object.
func (pp Properties) MarshalYAML() (any, error) {
	if pp == nil {
		return nil, nil
	}

	if len(pp) == 0 {
		return []any{}, nil
	}

	node := &yaml.Node{
		Kind: yaml.MappingNode,
	}

	for _, p := range pp {
		valueNode := &yaml.Node{}
		if err := valueNode.Encode(p.Value); err != nil {
			return nil, fmt.Errorf("yaml encode property value: %w", err)
		}

		node.Content = append(node.Content, &yaml.Node{
			Kind:  yaml.ScalarNode,
			Value: p.Key,
		}, valueNode)
	}

	return node, nil
}

// UnmarshalYAML implements yaml.Unmarshaler interface
// to unmarshal an object into a sorted list of properties.
func (pp *Properties) UnmarshalYAML(node *yaml.Node) error {
	d := len(node.Content)
	if d%2 != 0 {
		return fmt.Errorf("expected even items for key-value")
	}

	for i := 0; i < d; i += 2 {
		var prop Property

		keyNode := node.Content[i]
		if err := keyNode.Decode(&prop.Key); err != nil {
			return fmt.Errorf("yaml decode property key: %w", err)
		}

		valueNode := node.Content[i+1]
		if err := valueNode.Decode(&prop.Value); err != nil {
			return fmt.Errorf("yaml decode property value: %w", err)
		}

		*pp = append(*pp, prop)
	}

	return nil
}

func (pp Properties) AsMap() map[string]any {
	if pp == nil {
		return nil
	}

	out := map[string]any{}
	for _, p := range pp {
		if v, ok := out[p.Key]; ok {
			// If key already exists, we try to convert it to a slice
			// and append to it.
			if s, ok := v.([]any); ok {
				s = append(s, p.Value)
				out[p.Key] = s
			} else {
				out[p.Key] = []any{v, p.Value}
			}
		} else {
			out[p.Key] = p.Value
		}
	}
	return out
}

// Property key-value pair.
type Property struct {
	Key   string
	Value any
}
