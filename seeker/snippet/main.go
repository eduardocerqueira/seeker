//date: 2021-11-24T17:03:37Z
//url: https://api.github.com/gists/fdfd501775dad79b0e3b8d58dbe4a685
//owner: https://api.github.com/users/flavio

package main

import (
	_ "crypto/sha256"
	_ "crypto/sha512"
	"fmt"
	"github.com/docker/distribution/reference"
	"os"
)

type Image struct {
	Registry   string
	Repository string
	Tag        string
	Digest     string
}

func main() {
	if len(os.Args) != 2 {
		fmt.Println("Wrong number of args")
		os.Exit(1)
	}

	refStr := os.Args[1]
	named, err := reference.ParseNormalizedNamed(refStr)
	if err != nil {
		fmt.Printf("Cannot parse image name: %+v\n", err)
		os.Exit(1)
	}

	// Add the latest lag if they did not provide one.
	named = reference.TagNameOnly(named)

	i := Image{
		Registry:   reference.Domain(named),
		Repository: reference.Path(named),
	}

	// Add the tag if there was one.
	if tagged, ok := named.(reference.Tagged); ok {
		i.Tag = tagged.Tag()
	} else {
		i.Tag = "-"
	}

	// Add the digest if there was one.
	if canonical, ok := named.(reference.Canonical); ok {
		digest := canonical.Digest()
		i.Digest = string(digest)
	} else {
		i.Digest = "-"
	}

	fmt.Printf("%s - %+v", named, i)
}