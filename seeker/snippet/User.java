//date: 2023-04-26T16:56:21Z
//url: https://api.github.com/gists/24927d19873b805aa521e59e3b7194ae
//owner: https://api.github.com/users/CodeVaDOs

package com.datingon.entity.user;


import com.datingon.entity.BaseEntity;
import com.datingon.entity.chat.ChatRoom;
import com.datingon.entity.chat.Message;
import com.datingon.entity.grade.Grade;
import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.*;
import org.hibernate.annotations.OnDelete;
import org.hibernate.annotations.OnDeleteAction;

import javax.persistence.*;
import java.time.LocalDate;
import java.util.List;

@EqualsAndHashCode(callSuper = true)
@Entity
@Table(name = "users")
@Builder
@Data
@NoArgsConstructor
@AllArgsConstructor
public class User extends BaseEntity {
    private String password;

    @Column(unique = true)
    private String email;

    @Enumerated(EnumType.STRING)
    private Role role = Role.USER;

    @Column(name = "phone_number")
    private String phoneNumber;

    @Column(name = "full_name")
    private String fullName;

    @Column(name = "birthday")
    private LocalDate birthday;

    @Column(name = "country")
    private String country;

    @Column(name = "interests")
    private String interests;

    @Column(name = "about")
    private String about;

    @Column(name = "avatar_url")
    private String avatarUrl;

    @JsonIgnore
    @OneToMany(mappedBy = "sender", fetch = FetchType.LAZY)
    @OnDelete(action = OnDeleteAction.CASCADE)
    private List<Message> messageList;


    @JsonIgnore
    @ManyToMany(cascade = {
            CascadeType.PERSIST,
            CascadeType.MERGE
    })
    @JoinTable(name = "user_chat_rooms",
            uniqueConstraints = {@UniqueConstraint(columnNames={"user_id", "chat_room_id"})},
            joinColumns = @JoinColumn(name = "user_id"),
            inverseJoinColumns = @JoinColumn(name = "chat_room_id")
    )
    private List<ChatRoom> chatRoomList;


    @JsonIgnore
    @OneToMany(mappedBy = "userGiven", fetch = FetchType.LAZY)
    @OnDelete(action = OnDeleteAction.CASCADE)
    private List<Grade> gradesGiven;

    @JsonIgnore
    @OneToMany(mappedBy = "userReceived", fetch = FetchType.LAZY)
    @OnDelete(action = OnDeleteAction.CASCADE)
    private List<Grade> gradesReceived;



    public User(String password, String email) {
        this.password = "**********"
        this.email = email;
    }
}
