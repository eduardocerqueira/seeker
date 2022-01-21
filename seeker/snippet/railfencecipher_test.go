//date: 2022-01-21T16:47:49Z
//url: https://api.github.com/gists/eec6977f944a25e2612805e529016bcd
//owner: https://api.github.com/users/dhcgn

package railfencecipher

import (
	"testing"
)

func TestDecode(t *testing.T) {
	type args struct {
		s string
		n int
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "TestDecode",
			args: args{
				s: "WECRLTEERDSOEEFEAOCAIVDEN",
				n: 3,
			},
			want: "WEAREDISCOVEREDFLEEATONCE",
		},
		{
			name: "TestDecode Hello World",
			args: args{
				s: "Hoo!el,Wrdl l",
				n: 3,
			},
			want: "Hello, World!",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Decode(tt.args.s, tt.args.n); got != tt.want {
				t.Errorf("Decode() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestEncode(t *testing.T) {
	type args struct {
		s string
		n int
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "TestEncode",
			args: args{
				s: "WEAREDISCOVEREDFLEEATONCE",
				n: 3,
			},
			want: "WECRLTEERDSOEEFEAOCAIVDEN",
		},
		{
			name: "TestEncode Hello World",
			args: args{
				s: "Hello, World!",
				n: 3,
			},
			want: "Hoo!el,Wrdl l",
		},
		{
			name: "TestEncode 2",
			args: args{
				s: "WEAREDISCOVEREDFLEEATONCE",
				n: 2,
			},
			want: "WAEICVRDLETNEERDSOEEFEAOC",
		},
		{
			name: "TestEncode 4",
			args: args{
				s: "WEAREDISCOVEREDFLEEATONCE",
				n: 4,
			},
			want: "WIREEEDSEEEACAECVDLTNROFO",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Encode(tt.args.s, tt.args.n); got != tt.want {
				t.Errorf("Encode() = %v, want %v", got, tt.want)
			}
		})
	}
}
func TestDecodeInvalidParams(t *testing.T) {
	type args struct {
		s string
		n int
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "TestDecodeInvalidParams",
			args: args{
				s: "WECRLTEERDSOEEFEAOCAIVDEN",
				n: 1,
			},
			want: "",
		},
		{
			name: "TestDecodeInvalidParams Hello World",
			args: args{
				s: "",
				n: 3,
			},
			want: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Decode(tt.args.s, tt.args.n); got != tt.want {
				t.Errorf("Decode() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestEncodeInvalidParams(t *testing.T) {
	type args struct {
		s string
		n int
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "TestEncodeInvalidParams",
			args: args{
				s: "WECRLTEERDSOEEFEAOCAIVDEN",
				n: 1,
			},
			want: "",
		},
		{
			name: "TestEncodeInvalidParams Hello World",
			args: args{
				s: "",
				n: 3,
			},
			want: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Encode(tt.args.s, tt.args.n); got != tt.want {
				t.Errorf("Encode() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestEncodeDecode(t *testing.T) {
	type args struct {
		s string
		n int
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "TestEncode",
			args: args{
				s: "WEAREDISCOVEREDFLEEATONCE",
				n: 3,
			},
		},
		{
			name: "TestEncode Hello World",
			args: args{
				s: "Hello, World!",
				n: 3,
			},
		},
		{
			name: "TestEncode 2",
			args: args{
				s: "WEAREDISCOVEREDFLEEATONCE",
				n: 2,
			},
		},
		{
			name: "TestEncode 4",
			args: args{
				s: "WEAREDISCOVEREDFLEEATONCE",
				n: 4,
			},
		},
		{
			name: "TestEncode 12",
			args: args{
				s: "WEAREDISCOVEREDFLEEATONCE",
				n: 12,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoded := Encode(tt.args.s, tt.args.n)
			decoded := Decode(encoded, tt.args.n)

			if tt.args.s != decoded {
				t.Errorf("Encode() = input %v, decoded %v", tt.args.s, decoded)
			}
		})
	}
}

func BenchmarkDecode(b *testing.B) {
	for n := 0; n < b.N; n++ {
		Encode("WEAREDISCOVEREDFLEEATONCE", 3)
	}
}

func BenchmarkEncode(b *testing.B) {
	for n := 0; n < b.N; n++ {
		Encode("WECRLTEERDSOEEFEAOCAIVDEN", 3)
	}
}
