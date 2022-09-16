//date: 2022-09-16T22:11:49Z
//url: https://api.github.com/gists/89f75ac6e1f1de6e14686adb74665ba4
//owner: https://api.github.com/users/100mik

package main

import (
	"fmt"

	"sigs.k8s.io/kind/pkg/foo/version"
)

func main() {
	rawVersion := "v1.25.0"
	kubeVersion, err := version.ParseSemantic(rawVersion)
	if err != nil {
		fmt.Println("sigs/version be broke")
	}

	if kubeVersion.LessThan(version.MustParseSemantic("v1.24.0-alpha.1.592+370031cadac624")) {
		// for versions older than 1.24 prerelease remove only the old taint
		fmt.Println([]string{"node-role.kubernetes.io/master-"})
	} else if kubeVersion.LessThan(version.MustParseSemantic("v1.25.0-alpha.0.557+84c8afeba39ec9")) {
		// for versions between 1.24 and 1.25 prerelease remove both the old and new taint
		fmt.Println([]string{"node-role.kubernetes.io/control-plane-", "node-role.kubernetes.io/master-"})
	} else {
		// for any newer version only remove the new taint
		fmt.Println([]string{"node-role.kubernetes.io/control-plane-"})
	}
}
