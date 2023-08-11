//date: 2023-08-11T16:50:36Z
//url: https://api.github.com/gists/0a8564c9319580a711cbd1a127237374
//owner: https://api.github.com/users/debuggerpk

func WithVersionFromBuildInfo() ServiceOption {
	return func(s Service) {
		if info, ok := debug.ReadBuildInfo(); ok {
			var (
				revision  string
				modified  string
				timestamp time.Time
				version   string
			)

			for _, setting := range info.Settings {
				if setting.Key == "vcs.revision" {
					revision = setting.Value
				}

				if setting.Key == "vcs.modified" {
					modified = setting.Value
				}

				if setting.Key == "vcs.time" {
					timestamp, _ = time.Parse(time.RFC3339, setting.Value)
				}
			}

			if len(revision) > 0 && len(modified) > 0 && timestamp.Unix() > 0 {
				version = timestamp.Format("2006.01.02") + "." + revision[:8]
			} else {
				version = "debug"
			}

			if modified == "true" {
				version += "-dev"
			}

			s.(*config).Version = version
		}
	}
}