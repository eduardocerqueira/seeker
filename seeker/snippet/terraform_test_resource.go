//date: 2021-10-29T16:44:40Z
//url: https://api.github.com/gists/bd4935b84722175c34f4adcd05975730
//owner: https://api.github.com/users/hugorut

package aws_test

import (
	"testing"

	"github.com/infracost/infracost/internal/providers/terraform/tftest"
)

func TestTestResourceGoldenFile(t *testing.T) {
	t.Parallel()
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	tftest.GoldenFileResourceTests(t, "test_resource_test")
}
