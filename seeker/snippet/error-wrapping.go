//date: 2024-06-14T16:53:05Z
//url: https://api.github.com/gists/ed4bf198f9d4c306b3bcbe793f28f0a6
//owner: https://api.github.com/users/docsallover

type MyParseError struct {
  message string
  line    int
}

func (e *MyParseError) Error() string {
  return fmt.Sprintf("Parse error at line %d: %s", e.line, e.message)
}

func ParseConfig(data []byte) (*Config, error) {
  var cfg Config
  err := json.Unmarshal(data, &cfg)
  if err != nil {
    // Identify line number during parsing (assuming JSON data has line info)
    line := findErrorLine(data, err)
    return nil, &MyParseError{message: err.Error(), line: line}
  }
  return &cfg, nil
}