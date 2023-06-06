//date: 2023-06-06T17:06:55Z
//url: https://api.github.com/gists/58ffd20b1478ab7d5648c875db560261
//owner: https://api.github.com/users/ys2017rhein

package com.example.demo2.entity;

import lombok.Getter;
import lombok.Setter;

import javax.persistence.*;
import java.io.Serializable;

@Entity
@Table(name = "department")
@Getter
@Setter
public class Department implements Serializable {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private Long id;
    private String dname;
}
