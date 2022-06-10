//date: 2022-06-10T17:06:52Z
//url: https://api.github.com/gists/525a7c727ea2b1eda834267e572da175
//owner: https://api.github.com/users/yaq-cc

package getparams

import (
	"log"
	"net/url"
)

func AddGetParameters(base string) string {
	URL, err := url.Parse(base)
	if err != nil {
		log.Fatal(err)
	}
	query := URL.Query()
	query.Set("User", "Yvan")
	URL.RawQuery = query.Encode()
	return URL.String()
}