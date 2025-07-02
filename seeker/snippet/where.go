//date: 2025-07-02T17:02:42Z
//url: https://api.github.com/gists/1e6907c333b16e4edc0ef7418490e356
//owner: https://api.github.com/users/peteherman

package pagination

import (
	"fmt"
	"strings"
)

type clauseAndArgs struct {
	statement string
	args      []any
}

type whereClause struct {
	clauses []clauseAndArgs
}

func NewWhereClause() whereClause {
	return whereClause{
		clauses: []clauseAndArgs{},
	}
}

func (w *whereClause) AddClause(clause string, args ...any) {
	w.clauses = append(w.clauses, clauseAndArgs{
		statement: clause,
		args:      args,
	})
}

func (w *whereClause) ToConditionAndArgs(placeholderStartValue int) (string, []any) {

	allArgs := []any{}
	formattedClauses := make([]string, 0, len(w.clauses))

	placeholderValue := placeholderStartValue
	for _, clause := range w.clauses {
		placeholdersForClause := make([]int, 0, len(clause.args))
		for range clause.args {
			placeholdersForClause = append(placeholdersForClause, placeholderValue)
			placeholderValue++
		}
		allArgs = append(allArgs, clause.args...)

		clauseWithPlaceholders := fmt.Sprintf(clause.statement,
			convertIntSliceToAnySlice(placeholdersForClause)...)
		formattedClauses = append(formattedClauses, clauseWithPlaceholders)
	}

	condition := strings.Join(formattedClauses, " AND ")
	return condition, allArgs
}

func convertIntSliceToAnySlice(ints []int) []any {
	anySlice := make([]any, len(ints))
	for i, v := range ints {
		anySlice[i] = v
	}
	return anySlice

}
