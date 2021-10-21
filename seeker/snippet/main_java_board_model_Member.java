//date: 2021-10-21T17:01:51Z
//url: https://api.github.com/gists/f58dacba420e3ee28f5d9a38f5934d85
//owner: https://api.github.com/users/chacha86

package board.model;

public class Member {

	private int id;
	private String loginId;
	private String loginPw;
	private String nickname;
	private String regDate;
	
	public Member(int id, String loginId, String loginPw, String nickname, String regDate) {
		super();
		this.id = id;
		this.loginId = loginId;
		this.loginPw = loginPw;
		this.nickname = nickname;
		this.regDate = regDate;
	}
	
	public Member(String loginId, String loginPw, String nickname) {
		super();
		this.loginId = loginId;
		this.loginPw = loginPw;
		this.nickname = nickname;
	}
	
	public Member(String loginId, String loginPw) {
		super();
		this.loginId = loginId;
		this.loginPw = loginPw;
	}
	public Member() {
		
	}

	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}

	public String getLoginId() {
		return loginId;
	}

	public void setLoginId(String loginId) {
		this.loginId = loginId;
	}

	public String getLoginPw() {
		return loginPw;
	}

	public void setLoginPw(String loginPw) {
		this.loginPw = loginPw;
	}

	public String getNickname() {
		return nickname;
	}

	public void setNickname(String nickname) {
		this.nickname = nickname;
	}

	public String getRegDate() {
		return regDate;
	}

	public void setRegDate(String regDate) {
		this.regDate = regDate;
	}
	
}
