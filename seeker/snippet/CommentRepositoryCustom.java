//date: 2023-11-30T17:08:19Z
//url: https://api.github.com/gists/42cf15f7e82f395a729602c02f69b521
//owner: https://api.github.com/users/chaeeun-Han

package com.hamahama.pupmory.domain.memorial;

import com.hamahama.pupmory.domain.user.ServiceUser;

import java.util.List;

public interface CommentRepositoryCustom {
    List<Comment> findAllByUser(ServiceUser user);
}
