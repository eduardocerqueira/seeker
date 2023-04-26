//date: 2023-04-26T16:47:38Z
//url: https://api.github.com/gists/165fbcf59c7421ad0b4aff0d3e9db564
//owner: https://api.github.com/users/CodeVaDOs

package com.datingon.facade;

import com.datingon.dto.rq.GradeRequest;
import com.datingon.dto.rq.UserRequest;
import com.datingon.dto.rs.GradeResponse;
import com.datingon.dto.rs.UserResponse;
import com.datingon.entity.grade.Grade;
import com.datingon.entity.user.User;
import com.datingon.service.GradeService;
import com.datingon.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.security.Principal;

@Component
public class GradeFacade extends GeneralFacade<Grade, GradeRequest, GradeResponse> {
    @Autowired
    private UserService userService;

    @Autowired
    private GradeService service;

    @PostConstruct
    public void init() {
        super.getMm().typeMap(Grade.class, GradeResponse.class)
                .addMapping(Grade::getGradeType, GradeResponse::setGradeType)
                .addMapping(src -> src.getUserReceived().getId(), GradeResponse::setUserReceived);
    }

    public GradeResponse gradeUser(GradeRequest gradeRequest, Principal principal) {
        User userGiven = userService.getUserByEmail(principal.getName());
        User userReceived = userService.findEntityById(gradeRequest.getUserReceived());

        Grade grade = new Grade(gradeRequest.getGradeType(), userGiven, userReceived);

        return convertToDto(
                service.gradeUser(grade)
        );
    }
}
