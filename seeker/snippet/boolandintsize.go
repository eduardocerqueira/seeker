//date: 2022-04-12T17:07:26Z
//url: https://api.github.com/gists/2486254ce4f2b420a5691359f247dba5
//owner: https://api.github.com/users/oemegil

func main() {
   boolType := bool(false)
   int32Type := int32(44)
   fmt.Printf("bool type size : %d, int32 type size: %d \n", unsafe.Sizeof(boolType), unsafe.Sizeof(int32Type))
}
