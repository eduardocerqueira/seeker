//date: 2023-06-06T17:06:55Z
//url: https://api.github.com/gists/58ffd20b1478ab7d5648c875db560261
//owner: https://api.github.com/users/ys2017rhein

package com.example.demo2.entity;

import com.fasterxml.jackson.annotation.JsonBackReference;
import lombok.Getter;
import lombok.Setter;
import org.springframework.format.annotation.DateTimeFormat;

import javax.persistence.*;
import java.io.Serializable;
import java.util.Date;
import java.util.List;

@Entity
@Table(name = "user")
@Setter
@Getter
public class User implements Serializable {
    @Id
    @GeneratedValue(strategy=GenerationType.IDENTITY)
    private Long id;
    private String uname;
    @DateTimeFormat(pattern = "yy-MM-dd hh:mm:ss")
    private Date createDate;
    @ManyToOne
    @JoinColumn(name = "did")
    @JsonBackReference
    private Department department;

    @ManyToMany(cascade = {}, fetch = FetchType.EAGER)
    @JoinTable(name = "user_role",
    joinColumns = {@JoinColumn(name = "user_id")},
    inverseJoinColumns = {@JoinColumn(name = "role_id")})
    private List<Role> roles;
}
