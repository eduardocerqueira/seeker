//date: 2022-10-05T17:15:59Z
//url: https://api.github.com/gists/cb1eeb601f4b8e0c56d51da89423b24b
//owner: https://api.github.com/users/collectedview

package com.springreactgraphql.springreactgraphql.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import com.springreactgraphql.springreactgraphql.model.Blog;
import org.springframework.stereotype.Repository;

@Repository
public interface BlogRepository extends JpaRepository<Blog, Integer> {

    List<Blog> findByTitle(String title);

}