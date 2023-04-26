//date: 2023-04-26T16:56:21Z
//url: https://api.github.com/gists/24927d19873b805aa521e59e3b7194ae
//owner: https://api.github.com/users/CodeVaDOs

package com.datingon.entity;

import lombok.*;

import javax.persistence.*;
import java.io.Serializable;

import static javax.persistence.GenerationType.IDENTITY;

@MappedSuperclass
@Setter(AccessLevel.PUBLIC)
@Getter(AccessLevel.PUBLIC)
@EqualsAndHashCode(callSuper = true)
@Data
@NoArgsConstructor
@AllArgsConstructor
public class BaseEntity extends Auditable<String> implements Serializable {
    @Id
    @GeneratedValue(strategy = IDENTITY)
    @Column(name = "id", nullable = false, updatable = false)
    private Long id;

    @Version
    private Long version;
}