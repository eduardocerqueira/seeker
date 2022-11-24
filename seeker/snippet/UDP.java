//date: 2022-11-24T17:02:30Z
//url: https://api.github.com/gists/ba6867b242bfe69835edb3e4eb8b7837
//owner: https://api.github.com/users/zeddo123

public class UDP {
	static private int MIN_MTU = 576;
	static private int MAX_IP_HEADER_SIZE = 60;
	static private int UDP_HEADER_SIZE = 8;
	static public int MAX_DATAGRAM_SIZE = MIN_MTU - MAX_IP_HEADER_SIZE - UDP_HEADER_SIZE;
}