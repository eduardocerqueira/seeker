//date: 2021-10-28T16:53:46Z
//url: https://api.github.com/gists/7b5f3cbfc77c9130e1ce395541b940e2
//owner: https://api.github.com/users/ulexxander

const intSize = unsafe.Sizeof(int(0))

func intToUUID(n int) uuid.UUID {
	var u uuid.UUID
	intPtr := unsafe.Pointer(&n)
	for i := 0; i < int(intSize); i++ {
		byteptr := (*byte)(unsafe.Add(intPtr, i))
		u[i] = *byteptr
	}
	return u
}

func TestIntToUUID(t *testing.T) {
	tt := []struct {
		n int
		u uuid.UUID
	}{
		{
			n: 0,
			u: uuid.MustParse("00000000-0000-0000-0000-000000000000"),
		},
		{
			n: 5,
			u: uuid.MustParse("05000000-0000-0000-0000-000000000000"),
		},
		{
			n: 18,
			u: uuid.MustParse("12000000-0000-0000-0000-000000000000"),
		},
		{
			n: 255,
			u: uuid.MustParse("ff000000-0000-0000-0000-000000000000"),
		},
		{
			n: 256,
			u: uuid.MustParse("00010000-0000-0000-0000-000000000000"),
		},
		{
			n: 1024,
			u: uuid.MustParse("00040000-0000-0000-0000-000000000000"),
		},
	}

	for _, tc := range tt {
		t.Run(fmt.Sprint(tc.n), func(t *testing.T) {
			u := intToUUID(tc.n)
			if u.String() != tc.u.String() {
				t.Fatalf("expected %s got %s", tc.u, u)
			}
		})
	}
}