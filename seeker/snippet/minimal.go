//date: 2022-01-07T16:56:42Z
//url: https://api.github.com/gists/57ed59a6d57600c23913071b8470175b
//owner: https://api.github.com/users/wagoodman

package main

import (
	"fmt"

	"github.com/anchore/syft/syft"
	"github.com/anchore/syft/syft/format"
	"github.com/anchore/syft/syft/sbom"
	"github.com/anchore/syft/syft/source"
)

// use syft to discover packages + distro only
func main() {
	userInput := "ubuntu:latest"

	src, cleanup, err := source.New(userInput, nil, nil)
	if err != nil {
		panic(fmt.Errorf("failed to construct source from user input %q: %w", userInput, err))
	}
	if cleanup != nil {
		defer cleanup()
	}

	result := sbom.SBOM{
		Source: src.Metadata,
		// TODO: we should have helper functions for getting this built from exported library functions
		Descriptor: sbom.Descriptor{
			Name:    "syft",
			Version: "v-your-syft-version-here", // shows up in the output for many different formats
		},
	}

	packageCatalog, relationships, theDistro, err := syft.CatalogPackages(src, source.SquashedScope)
	if err != nil {
		panic(err)
	}

	result.Artifacts.PackageCatalog = packageCatalog
	result.Artifacts.Distro = theDistro
	result.Relationships = relationships

	// you can use other formats such as format.CycloneDxJSONOption or format.SPDXJSONOption ...
	bytes, err := syft.Encode(result, format.JSONOption)
	if err != nil {
		panic(err)
	}

	fmt.Println(string(bytes))
}
