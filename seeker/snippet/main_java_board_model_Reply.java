//date: 2021-10-21T17:01:51Z
//url: https://api.github.com/gists/f58dacba420e3ee28f5d9a38f5934d85
//owner: https://api.github.com/users/chacha86

package board.model;

public class Reply {
	private int id;
	private int articleId;
	private String body;
	private String nickname;
	private String regDate;
	private int memberId;
	
	public Reply(int id, int articleId, String body, String nickname, String regDate) {
		super();
		this.id = id;
		this.articleId = articleId;
		this.body = body;
		this.nickname = nickname;
		this.regDate = regDate;
	}
	public Reply(int articleId, String body, int memberId) {
		super();
		this.articleId = articleId;
		this.body = body;
		this.memberId = memberId;
	}
	
	public Reply(String body, String nickname, String regDate) {
		super();
		this.body = body;
		this.nickname = nickname;
		this.regDate = regDate;
	}
	
	public int getId() {
		return id;
	}
	public void setId(int id) {
		this.id = id;
	}
	public int getArticleId() {
		return articleId;
	}
	public void setArticleId(int articleId) {
		this.articleId = articleId;
	}
	public String getBody() {
		return body;
	}
	public void setBody(String body) {
		this.body = body;
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
	public int getMemberId() {
		return memberId;
	}
	public void setMemberId(int memberId) {
		this.memberId = memberId;
	}
	
}
