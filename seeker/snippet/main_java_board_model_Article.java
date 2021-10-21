//date: 2021-10-21T17:01:51Z
//url: https://api.github.com/gists/f58dacba420e3ee28f5d9a38f5934d85
//owner: https://api.github.com/users/chacha86

package board.model;

public class Article {
	
	private int id;
	private String title;
	private String body;
	private String regDate;
	private String nickname;
	private int rcnt;
	
	public Article(int id, String title, String body, String regDate, String nickname) {
		super();
		this.id = id;
		this.title = title;
		this.body = body;
		this.regDate = regDate;
		this.nickname = nickname;
	}

	public Article(String title, String body, String regDate, String nickname) {
		super();
		this.title = title;
		this.body = body;
		this.regDate = regDate;
		this.nickname = nickname;
	}

	public Article(int id, String title, String body, String regDate, String nickname, int rcnt) {
		super();
		this.id = id;
		this.title = title;
		this.body = body;
		this.regDate = regDate;
		this.nickname = nickname;
		this.rcnt = rcnt;
	}
	
	public Article(int id, String title, String body) {
		super();
		this.id = id;
		this.title = title;
		this.body = body;
	}
	
	public int getRcnt() {
		return rcnt;
	}

	public void setRcnt(int rcnt) {
		this.rcnt = rcnt;
	}
	
	public int getId() {
		return id;
	}
	public void setId(int id) {
		this.id = id;
	}
	public String getTitle() {
		return title;
	}
	public void setTitle(String title) {
		this.title = title;
	}
	public String getBody() {
		return body;
	}
	public void setBody(String body) {
		this.body = body;
	}
	public String getRegDate() {
		return regDate;
	}
	public void setRegDate(String regDate) {
		this.regDate = regDate;
	}
	public String getNickname() {
		return nickname;
	}
	public void setNickname(String nickname) {
		this.nickname = nickname;
	}
	
}
