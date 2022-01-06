//date: 2022-01-06T17:21:13Z
//url: https://api.github.com/gists/d203513edc2ef3d5dd28ea6a74bdf0f2
//owner: https://api.github.com/users/Integralist

// IsGlobalFlagsOnly indicates if the user called the binary with any
// permutation order of the globally defined flags.
//
// NOTE: Some global flags accept a value while others do not. The following
// algorithm takes this into account by mapping the flag to an expected value.
// For example, --verbose doesn't accept a value so is set to zero.
//
// EXAMPLES:
//
// The following would return false as a command was specified:
//
// args: [--verbose -v --endpoint ... --token ... -t ... --endpoint ...  version] 11
// total: 10
//
// The following would return true as only global flags were specified:
//
// args: [--verbose -v --endpoint ... --token ... -t ... --endpoint ...] 10
// total: 10
func IsGlobalFlagsOnly(args []string) bool {
	// Global flags are defined in pkg/app/run.go#84
	globals := map[string]int{
		"--verbose":  0,
		"-v":         0,
		"--token":    1,
		"-t":         1,
		"--endpoint": 1,
	}
	var total int
	for _, a := range args {
		for k := range globals {
			if a == k {
				total += 1
				total += globals[k]
			}
		}
	}
	return len(args) == total
}