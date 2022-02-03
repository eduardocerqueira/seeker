//date: 2022-02-03T17:08:00Z
//url: https://api.github.com/gists/c128157608f86c5f1be4a12b19ab71d0
//owner: https://api.github.com/users/nickistre

package utils

import (
	"reflect"
	"testing"
)

func TestSnakeCaseToCamelCase(t *testing.T) {
	type args struct {
		inputUnderScoreStr string
	}
	tests := []struct {
		name  string
		args  args
		want  string
		want1 int
	}{
		{
			name:  "Single word, all lower case",
			args:  args{"word"},
			want:  "word",
			want1: 1,
		},
		{
			name:  "Single word, mixed case",
			args:  args{"wOrD"},
			want:  "word",
			want1: 1,
		},
		{
			name:  "Multiple words, all lower case",
			args:  args{"one_two_three"},
			want:  "oneTwoThree",
			want1: 3,
		},
		{
			name:  "Multiple words, mixed case",
			args:  args{"One_tWo_thRee"},
			want:  "oneTwoThree",
			want1: 3,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1 := SnakeCaseToCamelCase(tt.args.inputUnderScoreStr)
			if got != tt.want {
				t.Errorf("SnakeCaseToCamelCase() = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("SnakeCaseToCamelCase() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func TestCamelCaseToSnakeCase(t *testing.T) {
	type args struct {
		inputCamelCaseStr string
	}
	tests := []struct {
		name  string
		args  args
		want  string
		want1 int
	}{
		{
			name:  "Single word",
			args:  args{"word"},
			want:  "word",
			want1: 1,
		},
		{
			name:  "Multiple words",
			args:  args{"twoWords"},
			want:  "two_words",
			want1: 2,
		},
		{
			name:  "Has underscore one word",
			args:  args{"one_word"},
			want:  "one_word",
			want1: 1,
		},
		{
			name:  "Has underscore multiple words",
			args:  args{"thisHas_underscore"},
			want:  "this_has_underscore",
			want1: 2,
		},
		{
			name:  "One word starts capitalized",
			args:  args{"Capitalized"},
			want:  "capitalized",
			want1: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1 := CamelCaseToSnakeCase(tt.args.inputCamelCaseStr)
			if got != tt.want {
				t.Errorf("CamelCaseToSnakeCase() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("CamelCaseToSnakeCase() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}
