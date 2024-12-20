//date: 2024-12-20T16:47:32Z
//url: https://api.github.com/gists/04e2a88cdb3fe6fafe2a1220f59a26b7
//owner: https://api.github.com/users/jspdown

func main() {
	var paths []string
	err := filepath.Walk("~/Oss/openapi-directory/APIs", func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			return fmt.Errorf("error accessing path %q: %v", path, err)
		}

		if info.IsDir() || !strings.HasSuffix(path, ".yaml") {
			return nil
		}

		paths = append(paths, path)

		return nil
	})
	if err != nil {
		panic(err)
	}

	stats := Stats{
		SchemaType: make(map[string]int),
		Properties: make(map[string]int),
	}
	uniqPatterns := make(map[string]struct{})
	uniqEnums := make(map[string]struct{})

	for _, path := range paths {

		loader := openapi3.NewLoader()
		doc, err := loader.LoadFromFile(path)
		if err != nil {
			log.Println("Unable to load specification", path, err)
			continue
		}

		var params []openapi3.Parameter
		for _, pathItem := range doc.Paths {
			for _, param := range pathItem.Parameters {
				if param.Value.In != openapi3.ParameterInPath {
					continue
				}

				params = append(params, *param.Value)
			}

			for _, operation := range pathItem.Operations() {
				for _, param := range operation.Parameters {
					if param.Value.In != openapi3.ParameterInPath {
						continue
					}

					params = append(params, *param.Value)
				}
			}
		}

		stats.Total += len(params)
		for _, param := range params {
			if param.Schema == nil {
				continue
			}

			s := param.Schema.Value

			stats.WithSchema++
			stats.SchemaType[s.Type]++

			props := map[string]bool{
				// Complex shit.
				"oneOf": s.OneOf != nil,
				"anyOf": s.AnyOf != nil,
				"allOf": s.AllOf != nil,
				"not":   s.Not != nil,
				// Simple.
				"enum":   s.Enum != nil,
				"format": s.Format != "",
				// Properties.
				"exclusiveMin":    s.ExclusiveMin,
				"exclusiveMax":    s.ExclusiveMax,
				"nullable":        s.Nullable,
				"readInly":        s.ReadOnly,
				"writeOnly":       s.WriteOnly,
				"allowEmptyValue": s.AllowEmptyValue,
				// Number.
				"min":        s.Min != nil,
				"max":        s.Max != nil,
				"multipleOf": s.MultipleOf != nil,
				// String.
				"minLength": s.MinLength != 0,
				"maxLength": s.MaxLength != nil,
				"pattern":   s.Pattern != "",
				// Array
				"minItems": s.MinItems != 0,
				"maxItems": s.MaxItems != nil,
				"items":    s.Items != nil,
				// Object
				"required":             s.Required != nil,
				"properties":           s.Properties != nil,
				"minProps":             s.MinProps != 0,
				"maxProps":             s.MaxProps != nil,
				"additionalProperties": s.AdditionalProperties.Has != nil,
				"discriminator":        s.Discriminator != nil,
			}

			for name, isSet := range props {
				if isSet {
					stats.Properties[name]++
				}
			}

			if s.Pattern != "" {
				if _, ok := uniqPatterns[s.Pattern]; !ok {
					uniqPatterns[s.Pattern] = struct{}{}
					stats.Patterns = append(stats.Patterns, s.Pattern)
				}
			}
			if s.Enum != nil {
				key := fmt.Sprintf("%q", s.Enum)
				if _, ok := uniqEnums[key]; !ok {
					uniqEnums[key] = struct{}{}
					stats.Enums = append(stats.Enums, s.Enum)
				}
			}
		}
	}

	spew.Dump(stats)
}
