//date: 2024-11-19T17:11:11Z
//url: https://api.github.com/gists/9fdd16fb5d2d8e610e0a6f3fd1eb28a0
//owner: https://api.github.com/users/jclem

func (c Collection) Validate() error {
    v := vld.New()
    strlen := vld.StrLen(1, 64)
    
    v.Validate(
        []string{"name"},
        strlen(c.Name),
    )

    v.Validate(
        []string{"tags", "key"},
        vld.MapKeys[string, string](strlen)(c.Tags),
    )

    v.Validate(
        []string{"tags", "value"},
        vld.MapValues[string](strlen)(c.Tags),
    )

    if err := v.GetError(); err != nil {
        return fmt.Errorf("validate collection: %w", err)
    }

    return nil
}