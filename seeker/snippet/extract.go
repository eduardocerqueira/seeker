//date: 2024-09-02T17:05:23Z
//url: https://api.github.com/gists/8eca49b36dab6eda8163569ff0a1ab66
//owner: https://api.github.com/users/pjmagee

package main

import (
	"context"
	"dagger/dsg-icis-openapi-kiota/internal/dagger"
	"fmt"
	"slices"
)

type Language string

const (
	CSharp string = "CSharp"
	Go     string = "Go"
	Java   string = "Java"
	Python string = "Python"
)

var Specs = []Spec{
	NewSpec("https://developer.icis.com/portals/api/sites/icis-live-portal/liveportal/apis/energyapi/download_spec", []Settings{
		NewSettings(CSharp, "Icis.Api.Energy", "Energy"),
		NewSettings(Go, "icis/api/energy", "energy"),
		NewSettings(Java, "com.icis.api.energy", "src/main/java/com/icis/api/energy"),
		NewSettings(Python, "icis_api_energy", "energy"),
	}),
	NewSpec("https://developer.icis.com/portals/api/sites/icis-live-portal/liveportal/apis/energyforesightapi/download_spec", []Settings{
		NewSettings(CSharp, "Icis.Api.EnergyForesight", "EnergyForesight"),
		NewSettings(Go, "icis/api/energyforesight", "energyforesight"),
		NewSettings(Java, "com.icis.api.energyforesight", "src/main/java/com/icis/api/energyforesight"),
		NewSettings(Python, "icis_api_energyforesight", "energyforesight"),
	}),
	NewSpec("https://developer.icis.com/portals/api/sites/icis-live-portal/liveportal/apis/lnganalyticsapi/download_spec", []Settings{
		NewSettings(CSharp, "Icis.Api.LngAnalytics", "LngAnalytics"),
		NewSettings(Go, "icis/api/lnganalytics", "lnganalytics"),
		NewSettings(Java, "com.icis.api.lnganalytics", "src/main/java/com/icis/api/lnganalytics"),
		NewSettings(Python, "icis_api_lnganalytics", "lnganalytics"),
	}),
}

type Settings struct {
	Language  string
	Namespace string
	Path      string
}

func NewSettings(language string, namespace string, path string) Settings {
	return Settings{
		Language:  language,
		Namespace: namespace,
		Path:      path,
	}
}

type Spec struct {
	URL      string
	Settings []Settings
}

func NewSpec(url string, settings []Settings) Spec {
	return Spec{
		URL:      url,
		Settings: settings,
	}
}

// Generates Kiota clients for the ICIS OpenAPI specs
func (m *DsgIcisOpenapiKiota) GenerateKiotaClients(
	ctx context.Context,
	languages []string,
	// +optional
	// +default="1.15.0"
	// The default version of the Kiota tool to use
	version string) *dagger.Container {

	return dag.
		Container().
		From("mcr.microsoft.com/dotnet/sdk:8.0").
		WithoutUser().
		WithExec([]string{"dotnet", "tool", "install", "Microsoft.OpenApi.Kiota", "--tool-path", "/app"}).
		WithWorkdir("/app").
		WithoutEntrypoint().
		With(func(r *dagger.Container) *dagger.Container {
			return kiotaCommands(r, languages)
		}).
		With(func(r *dagger.Container) *dagger.Container {
			entries, _ := r.Directory("/output").Entries(ctx)
			if len(entries) > 0 {
				return r.WithExec([]string{"tar", "-czvf", "/output.tar.gz", "-C", "/", "output"})
			}
			return r
		})
}

// func (m *DsgIcisOpenapiKiota) Kiota(version string) *dagger.Container {
// 	return dag.
// 		Container().
// 		WithoutUser().
// 		From("mcr.microsoft.com/dotnet/sdk:8.0").
// 		WithExec([]string{"mkdir", "/app"}).
// 		WithExec([]string{"mkdir", "/output"}).
// 		WithExec([]string{"dotnet", "tool", "install", "Microsoft.OpenApi.Kiota", "--tool-path", "/app", "--version", version})
// }

func kiotaCommands(container *dagger.Container, languages []string) *dagger.Container {

	for _, spec := range Specs {
		for _, settings := range spec.Settings {
			if slices.Contains(languages, settings.Language) {
				container = container.
					WithExec([]string{
						"./kiota",
						"generate",
						"--output", fmt.Sprintf("/output/%s/%s", settings.Language, settings.Path),
						"--language", string(settings.Language),
						"--namespace-name", settings.Namespace,
						"--openapi", spec.URL,
						"--exclude-backward-compatible", "true",
						"--log-level", "Debug",
						"--additional-data", "true",
						"--class-name", "ApiClient",
					})
			}
		}
	}

	return container
}