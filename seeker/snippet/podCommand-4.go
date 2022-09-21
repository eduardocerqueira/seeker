//date: 2022-09-21T17:03:36Z
//url: https://api.github.com/gists/bd5d3020ac3f99b4363b543750bba664
//owner: https://api.github.com/users/26tanishabanik

i := 0
flag := 0
reader := bufio.NewReader(stdout)
for {
  strline, err := readLine(reader)
  if err != nil && err != io.EOF {
    log.Println(err)
  }
  if len(strline) > 0 {
    words:= strings.Fields(strline)
    portShown:= strings.Split(words[3], ":")
    if i >0 {
      givenPort,_ := strconv.ParseInt(port, 10,0)
      shownPort, _ := strconv.ParseInt(portShown[1], 10, 0)
      if givenPort == shownPort {
        flag = 1
        break
      }
    }	
  }
  i += 1
  if err == io.EOF {
    break
  }
}
if flag == 1{
  fmt.Printf("%s is open and listening", port)
}else{
  fmt.Printf("%s is closed", port)
}