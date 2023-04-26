//date: 2023-04-26T16:56:21Z
//url: https://api.github.com/gists/24927d19873b805aa521e59e3b7194ae
//owner: https://api.github.com/users/CodeVaDOs

package com.datingon.entity;

import lombok.Getter;
import lombok.Setter;
import org.springframework.data.annotation.CreatedBy;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.LastModifiedBy;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import javax.persistence.Column;
import javax.persistence.EntityListeners;
import javax.persistence.MappedSuperclass;
import java.util.Date;

@Getter
@Setter
@MappedSuperclass
@EntityListeners(AuditingEntityListener.class)
public class Auditable<U> {
    @CreatedBy
    @Column(name = "created_by")
    private U createdBy;

    @CreatedDate
    @Column(name = "created_date")
    private Date createdDate;

    @LastModifiedBy
    @Column(name = "updated_by")
    private U updatedBy;

    @LastModifiedDate
    @Column(name = "updated_date")
    private Date updatedDate;
}