//date: 2023-11-30T17:08:19Z
//url: https://api.github.com/gists/42cf15f7e82f395a729602c02f69b521
//owner: https://api.github.com/users/chaeeun-Han

package com.hamahama.pupmory.domain.memorial;

import com.hamahama.pupmory.domain.user.ServiceUser;
import com.querydsl.jpa.impl.JPAQueryFactory;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
@RequiredArgsConstructor
public class CommentRepositoryImpl implements CommentRepositoryCustom {

    private final JPAQueryFactory queryFactory;

    public List<Comment> findAllByUser(ServiceUser user) {
        // SELECT * FROM comment WHERE user_uid =

        QComment comment = QComment.comment;
        QPost post = QPost.post;

        return queryFactory.selectFrom(comment)
                .leftJoin(comment.post, post)
                .fetchJoin()
                .where(comment.user.eq(user))
                .fetch();
    }
}
