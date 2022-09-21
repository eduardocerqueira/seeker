//date: 2022-09-21T17:05:11Z
//url: https://api.github.com/gists/6d14782708c2f47f1c46fffd9bf3cb9a
//owner: https://api.github.com/users/26tanishabanik

func main(){
	if len(os.Args) < 3{
		fmt.Println("First argument is for pod name, second argument is for port number and third argument is for namespace")
	}else {
		PodCommand(os.Args[1],os.Args[2], os.Args[3])
	}
}