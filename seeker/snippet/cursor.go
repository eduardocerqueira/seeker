//date: 2025-07-02T17:02:42Z
//url: https://api.github.com/gists/1e6907c333b16e4edc0ef7418490e356
//owner: https://api.github.com/users/peteherman

package pagination

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/peteherman/iwasthere/internal/errors"
	"github.com/peteherman/iwasthere/internal/sort"
)

type CursorPagination struct {
	Next *string
}

type sortPairWithLastColumnValue[T ~string] struct {
	SortPair[T]
	LastSortColumnValue any `json:"v"`
}

type SortPair[T ~string] struct {
	SortColumn T          `json:"c"`
	Direction  sort.Order `json:"d"`
}

type cursor[T ~string] struct {
	validated bool

	LastRecordInternalId int64 `json:"i"`

	InternalIdColumnName T `json:"n"`

	SortPairs []sortPairWithLastColumnValue[T] `json:"p"`
}

func NewCursor[T ~string](internalIdColumnName T, lastRecordInternalId int64) *cursor[T] {
	return &cursor[T]{
		LastRecordInternalId: lastRecordInternalId,
		InternalIdColumnName: internalIdColumnName,
		validated:            false,
		SortPairs:            make([]sortPairWithLastColumnValue[T], 0),
	}
}

var _ Cursor[string] = &cursor[string]{}

type Cursor[T ~string] interface {
	IsEmpty() bool
	ToString() (string, error)
	IsValid() bool
	AddSortPair(SortPair[T], any)
	ToQueryClausesAndArguments() (CursorQueryClausesAndArguments, error)
}

func (c *cursor[T]) IsEmpty() bool {
	return c.LastRecordInternalId == 0
}

func (c *cursor[T]) ToString() (string, error) {
	jsonBytes, err := json.Marshal(c)
	if err != nil {
		return "", err
	}
	return base64.URLEncoding.EncodeToString(jsonBytes), nil
}

func (c *cursor[T]) IsValid() bool {
	return c.validated
}

func (c *cursor[T]) AddSortPair(pair SortPair[T], lastSortColumnValue any) {

	pairWithLastColumnValue := sortPairWithLastColumnValue[T]{
		SortPair:            pair,
		LastSortColumnValue: lastSortColumnValue,
	}

	c.SortPairs = append(c.SortPairs, pairWithLastColumnValue)
}

type CursorQueryClausesAndArguments struct {
	WhereCondition whereClause
	OrderStatement string
}

func (c *cursor[T]) ToQueryClausesAndArguments() (CursorQueryClausesAndArguments, error) {
	if !c.IsValid() {
		return CursorQueryClausesAndArguments{},
			fmt.Errorf("attempting to use unvalidated cursor")
	}

	whereCond := NewWhereClause()
	orderingClauses := []string{}
	for _, sortPair := range c.SortPairs {
		direction := ">"
		if sortPair.Direction == sort.Descending {
			direction = "<"
		}

		clause := fmt.Sprintf("(%s %s $%%d OR (%s = $%%d AND %s = $%%d))",
			sortPair.SortColumn,
			direction,
			sortPair.SortColumn,
			c.InternalIdColumnName,
		)
		whereCond.AddClause(
			clause,
			sortPair.LastSortColumnValue,
			sortPair.LastSortColumnValue,
			c.LastRecordInternalId,
		)

		orderClause := fmt.Sprintf("%s %s", sortPair.SortColumn, string(sortPair.Direction))
		orderingClauses = append(orderingClauses, orderClause)
	}

	// Lastly, add internal id as final pagination check
	orderingClauses = append(orderingClauses, fmt.Sprintf("%s ASC", c.InternalIdColumnName))

	orderStatement := strings.Join(orderingClauses, ", ")

	return CursorQueryClausesAndArguments{
		WhereCondition: whereCond,
		OrderStatement: orderStatement,
	}, nil
}

func SafeDecode[T ~string](s string, allowedSortColumnNames []T) (*cursor[T], error) {
	decoded, err := base64.URLEncoding.DecodeString(s)
	if err != nil {
		return nil, err
	}

	var res cursor[T]
	if err := json.Unmarshal(decoded, &res); err != nil {
		return nil, errors.ErrInvalidPaginationToken
	}

	allowedColumnNameSet := setOfAllowedSortColumnNames(allowedSortColumnNames)
	for _, sortPair := range res.SortPairs {
		if _, exists := allowedColumnNameSet[sortPair.SortColumn]; !exists {
			return nil, errors.ErrInvalidPaginationToken
		}
	}

	res.validated = true

	return &res, nil
}

func setOfAllowedSortColumnNames[T ~string](names []T) map[T]struct{} {
	res := make(map[T]struct{}, len(names))
	for _, name := range names {
		res[name] = struct{}{}
	}
	return res
}
