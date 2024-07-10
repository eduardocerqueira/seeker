//date: 2024-07-10T16:57:59Z
//url: https://api.github.com/gists/1e4fa5d0eb935071145be06d4af45ee1
//owner: https://api.github.com/users/BK1031

func main() {
	ConnectMQTT()
	SubscribeECU()
	SubscribeBattery()
	ConnectDB()
	StartServer()
}