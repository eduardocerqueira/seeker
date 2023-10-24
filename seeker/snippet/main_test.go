//date: 2023-10-24T16:34:28Z
//url: https://api.github.com/gists/45b2df58de8271d1ecc22fb97034fb40
//owner: https://api.github.com/users/zacharysyoung

package main

import (
	"fmt"
	"reflect"
	"testing"
)

var groupsTable = []struct {
	code string     // the observed code, i.e., "10"
	want [][]string // groups of digits adjacent to each digit from code, i.e., "10" -> {{"1","2","4"},{"0","8"}}
}{
	{"5", [][]string{
		{"2", "4", "5", "6", "8"},
	}},
	{"10", [][]string{
		{"1", "2", "4"},
		{"0", "8"},
	}},
	{"46", [][]string{
		{"1", "4", "5", "7"},
		{"3", "5", "6", "9"},
	}},
	{"123", [][]string{
		{"1", "2", "4"},
		{"1", "2", "3", "5"},
		{"2", "3", "6"},
	}},
	{"1357", [][]string{
		{"1", "2", "4"},
		{"2", "3", "6"},
		{"2", "4", "5", "6", "8"},
		{"4", "7", "8"},
	}},
}

var resultsTable = []struct {
	code string   // the observed code, i.e., "10"
	want []string // list of possible, sorted codes, i.e., "5" -> {"10","18","20","28","40","48"}
}{
	{"5", []string{
		"2", "4", "5", "6", "8",
	}},
	{"10", []string{
		"10", "18",
		"20", "28",
		"40", "48",
	}},
	{"46", []string{
		"13", "15", "16", "19",
		"43", "45", "46", "49",
		"53", "55", "56", "59",
		"73", "75", "76", "79",
	}},
	{"123", []string{
		"112", "113", "116", "122", "123", "126", "132", "133", "136", "152", "153", "156",
		"212", "213", "216", "222", "223", "226", "232", "233", "236", "252", "253", "256",
		"412", "413", "416", "422", "423", "426", "432", "433", "436", "452", "453", "456",
	}},
	{"1357", []string{
		"1224", "1227", "1228", "1244", "1247", "1248", "1254", "1257", "1258", "1264", "1267", "1268", "1284", "1287", "1288", "1324", "1327", "1328", "1344", "1347", "1348", "1354", "1357", "1358", "1364", "1367", "1368", "1384", "1387", "1388", "1624", "1627", "1628", "1644", "1647", "1648", "1654", "1657", "1658", "1664", "1667", "1668", "1684", "1687", "1688",
		"2224", "2227", "2228", "2244", "2247", "2248", "2254", "2257", "2258", "2264", "2267", "2268", "2284", "2287", "2288", "2324", "2327", "2328", "2344", "2347", "2348", "2354", "2357", "2358", "2364", "2367", "2368", "2384", "2387", "2388", "2624", "2627", "2628", "2644", "2647", "2648", "2654", "2657", "2658", "2664", "2667", "2668", "2684", "2687", "2688",
		"4224", "4227", "4228", "4244", "4247", "4248", "4254", "4257", "4258", "4264", "4267", "4268", "4284", "4287", "4288", "4324", "4327", "4328", "4344", "4347", "4348", "4354", "4357", "4358", "4364", "4367", "4368", "4384", "4387", "4388", "4624", "4627", "4628", "4644", "4647", "4648", "4654", "4657", "4658", "4664", "4667", "4668", "4684", "4687", "4688",
	}},
}

func TestGroups(t *testing.T) {
	for _, tc := range groupsTable {
		got := getGroups(tc.code)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("getGroups(%s) = %v; want %v\n", tc.code, got, tc.want)
		}
	}
}

func TestRecurse(t *testing.T) {
	for _, tc := range resultsTable {
		groups := getGroups(tc.code)
		got := recurse(groups)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("recurse(%s) = %v; want %v\n", groups, got, tc.want)
		}
	}
}
func TestProduct(t *testing.T) {
	for _, tc := range resultsTable {
		groups := getGroups(tc.code)
		got := product(groups)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("product(%s) = %v; want %v\n", groups, got, tc.want)
		}
	}
}

func Benchmark(b *testing.B) {
	for _, tc := range groupsTable {
		groups := tc.want

		b.Run(fmt.Sprintf("Recurse_%s", tc.code), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				recurse(groups)
			}
		})

		b.Run(fmt.Sprintf("Product_%s", tc.code), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				product(groups)
			}
		})
	}
}
