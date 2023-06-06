//date: 2023-06-06T17:06:55Z
//url: https://api.github.com/gists/58ffd20b1478ab7d5648c875db560261
//owner: https://api.github.com/users/ys2017rhein

package com.example.demo2.repository;

import com.example.demo2.entity.Department;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface DepartmentRepository extends JpaRepository<Department, Long> {
}
