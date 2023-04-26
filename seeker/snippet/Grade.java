//date: 2023-04-26T16:56:21Z
//url: https://api.github.com/gists/24927d19873b805aa521e59e3b7194ae
//owner: https://api.github.com/users/CodeVaDOs

package com.datingon.entity.grade;

import com.datingon.entity.BaseEntity;
import com.datingon.entity.user.User;
import lombok.*;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;

import javax.persistence.*;

@EqualsAndHashCode(callSuper = true)
@Entity
@Table(name = "grades")
@Builder
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Grade extends BaseEntity {
    @Enumerated(EnumType.STRING)
    private GradeType gradeType;

    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "user_given_id", referencedColumnName = "id")
    private User userGiven;

    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "user_received_id", referencedColumnName = "id")
    private User userReceived;
}
