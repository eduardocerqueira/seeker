//date: 2021-10-21T17:01:51Z
//url: https://api.github.com/gists/f58dacba420e3ee28f5d9a38f5934d85
//owner: https://api.github.com/users/chacha86

package board.model;

import java.util.List;

public interface ArticleMapper {

	void insertArticle(Article a);
	List<Article> selectArticle();
	void updateArticle(Article a);
	void deleteArticle(int id);
	List<Article> selectArticleByKeyword(String keyword);
	Article selectArticleById(int id);
	void insertMember(Member m);
	Member selectMemberByLoginId(String loginId);
	Member selectMemberByLoginIdAndLoginPw(Member m);
	void insertReply(Reply r);
	List<Reply> selectRepliesByArticleId(int id);

}
