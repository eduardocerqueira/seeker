//date: 2023-07-20T17:02:14Z
//url: https://api.github.com/gists/a920dcfea6cca057e517443c8a9423c0
//owner: https://api.github.com/users/delveper

package tmpl

import (
	"embed"
	"fmt"
	"text/template"
)

var (
	//go:embed *.tmpl
	fS embed.FS
)

const (
	TemplateEmail        = "email"
	TemplateEmailSubject = "subject"
	TemplateEmailBody    = "body"
)

// Load parses templates in the current directory.
func Load() (*template.Template, error) {
	// TODO: improve flexibility to choose templates.
	tmpl, err := template.ParseFS(fS, "*")
	if err != nil {
		return nil, fmt.Errorf("parsing templates: %w", err)
	}

	return tmpl, nil
}