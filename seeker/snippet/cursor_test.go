//date: 2025-07-02T17:02:42Z
//url: https://api.github.com/gists/1e6907c333b16e4edc0ef7418490e356
//owner: https://api.github.com/users/peteherman

package pagination_test

import (
	"testing"

	"github.com/peteherman/iwasthere/internal/pagination"
	"github.com/peteherman/iwasthere/internal/sort"
	"github.com/stretchr/testify/require"
)

type testSortColumn string

func TestCursor(t *testing.T) {
	t.Parallel()
	t.Run("IsValid", func(t *testing.T) {
		t.Parallel()
		t.Run("invalid", func(t *testing.T) {
			t.Parallel()
			myCursor := pagination.NewCursor[testSortColumn]("col_name", 0)
			require.False(t, myCursor.IsValid())
		})

		t.Run("valid", func(t *testing.T) {
			t.Parallel()
			myCursor := pagination.NewCursor[testSortColumn]("col_name", 0)
			myCursor.AddSortPair(pagination.SortPair[testSortColumn]{
				SortColumn: testSortColumn("test"),
				Direction:  sort.Ascending,
			}, 1)

			cursorStr, err := myCursor.ToString()
			require.NoError(t, err)

			decoded, err := pagination.SafeDecode(
				cursorStr,
				[]testSortColumn{testSortColumn("test")},
			)
			require.NoError(t, err)

			require.True(t, decoded.IsValid())
		})
	})

	t.Run("SafeDecode", func(t *testing.T) {
		t.Parallel()
		t.Run("invalid", func(t *testing.T) {
			myCursor := pagination.NewCursor[testSortColumn]("col_name", 0)
			myCursor.AddSortPair(pagination.SortPair[testSortColumn]{

				SortColumn: testSortColumn("test"),
				Direction:  sort.Ascending,
			}, 1)

			cursorStr, err := myCursor.ToString()
			require.NoError(t, err)

			_, err = pagination.SafeDecode(
				cursorStr,
				[]testSortColumn{testSortColumn("nope")},
			)
			require.Error(t, err)
		})
	})
}
