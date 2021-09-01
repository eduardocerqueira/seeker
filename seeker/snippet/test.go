//date: 2021-09-01T01:38:13Z
//url: https://api.github.com/gists/a89da3a41505d122c85228062a4e5a8e
//owner: https://api.github.com/users/meirelles

package pagination

import (
	"merchant-services/pkg/global"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPagination(t *testing.T) {
	tests := []struct {
		name           string
		perPage        int
		curPage        int
		totalItems     int
		expected       Pagination
		expectedOffset int
		expectedErr    string
	}{
		{
			name:        "invalid per page",
			perPage:     -100,
			curPage:     0,
			totalItems:  100,
			expectedErr: "-100 isn't a valid pagination limit number",
		},
		{
			name:        "negative cur page",
			perPage:     50,
			curPage:     -10,
			totalItems:  100,
			expectedErr: "-10 isn't a valid pagination page number",
		},
		{
			name:        "invalid cur page, greater than total pages",
			perPage:     10,
			curPage:     11,
			totalItems:  99,
			expectedErr: "out of bounds page 11 (max 10)",
		},
		{
			name:        "out of bounds non-zero when totalItems == 0",
			perPage:     0,
			curPage:     2,
			totalItems:  0,
			expectedErr: "out of bounds page 2 (max 0)",
		},
		{
			name:       "out of bounds but page = 1, so no error",
			perPage:    0,
			curPage:    1,
			totalItems: 0,
			expected: Pagination{
				TotalCount:    0,
				TotalPages:    0,
				CountReturned: 0,
				Page:          0,
				Limit:         50,
			},
		},
		{
			name:       "zero items (no error first page)",
			perPage:    0,
			curPage:    1,
			totalItems: 0,
			expected: Pagination{
				TotalCount:    0,
				TotalPages:    0,
				Page:          0,
				CountReturned: 0,
				Limit:         50,
			},
		},
		{
			name:       "one item",
			perPage:    0,
			curPage:    0,
			totalItems: 1,
			expected: Pagination{
				TotalCount:    1,
				TotalPages:    1,
				Page:          1,
				CountReturned: 1,
				Limit:         50,
			},
		},
		{
			name:       "15 items, 2nd page",
			perPage:    10,
			curPage:    2,
			totalItems: 15,
			expected: Pagination{
				TotalCount:    15,
				TotalPages:    2,
				Page:          2,
				CountReturned: 5,
				Limit:         10,
			},
			expectedOffset: 10,
		},
		{
			name:       "11 items, 2nd page",
			perPage:    10,
			curPage:    2,
			totalItems: 11,
			expected: Pagination{
				TotalCount:    11,
				TotalPages:    2,
				Page:          2,
				CountReturned: 1,
				Limit:         10,
			},
			expectedOffset: 10,
		},
		{
			name:       "3 items, 1 per page, 2nd page",
			perPage:    1,
			curPage:    3,
			totalItems: 3,
			expected: Pagination{
				TotalCount:    3,
				TotalPages:    3,
				Page:          3,
				CountReturned: 1,
				Limit:         1,
			},
			expectedOffset: 2,
		},
		{
			name:       "valid pagination, last one",
			perPage:    10,
			curPage:    10,
			totalItems: 99,
			expected: Pagination{
				TotalCount:    99,
				TotalPages:    10,
				CountReturned: 9,
				Page:          10,
				Limit:         10,
			},
			expectedOffset: 90,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := NewPagination(test.perPage, test.curPage, test.totalItems)

			global.CheckError(t, test.expectedErr, err)

			assert.Equal(t, test.expected, got)
			assert.Equal(t, test.expectedOffset, got.Offset())
		})
	}
}
