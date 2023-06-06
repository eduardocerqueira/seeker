//date: 2023-06-06T17:06:55Z
//url: https://api.github.com/gists/58ffd20b1478ab7d5648c875db560261
//owner: https://api.github.com/users/ys2017rhein

package com.example.demo2;

import com.example.demo2.entity.Department;
import com.example.demo2.entity.Role;
import com.example.demo2.entity.User;
import com.example.demo2.repository.DepartmentRepository;
import com.example.demo2.repository.RoleRepository;
import com.example.demo2.repository.UserRepository;
import config.JpaConfiguration;
import javafx.application.Application;
import org.junit.Before;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.util.Assert;


import java.util.Date;
import java.util.List;

//@SpringBootTest
@SpringBootTest(classes = Demo2Application.class)//属性用于指定引导类
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(classes = {JpaConfiguration.class})
class Demo2ApplicationTests {

    private static Logger logger = LoggerFactory.getLogger(Demo2Application.class);

    @Autowired
    private UserRepository userRepository;
    @Autowired
    private DepartmentRepository departmentRepository;
    @Autowired
    private RoleRepository roleRepository;

    @BeforeEach
    public void initData() {
        userRepository.deleteAll();
        departmentRepository.deleteAll();
        roleRepository.deleteAll();

        Department department = new Department();
        department.setDname("开发部");
        departmentRepository.save(department);
        Assert.notNull(department.getId(), "department的id不能为空");

        Role role = new Role();
        role.setRname("admin");
        roleRepository.save(role);
        Assert.notNull(role.getId(), "role的id不能为空");

        User user = new User();
        user.setUname("user");
        user.setCreateDate(new Date());
        user.setDepartment(department);
        List<Role> roles = roleRepository.findAll();
        user.setRoles(roles);
        userRepository.save(user);
        Assert.notNull(user.getId(), "user的id不能为空");
    }

    @Test
    public void test() {
//        userRepository.findAll();//
        //分页查询
        Pageable pageable = PageRequest.of(0,5, Sort.by(Sort.Direction.ASC, "id"));
        Page<User> userPage = userRepository.findAll(pageable);
        Assert.notNull(userPage, "userPage不能为空");
        for (User user1 :
                userPage.getContent()) {
            logger.info("====user=== user name:{}, department name:{}, role name:{}",
                    user1.getUname(), user1.getDepartment().getDname(), user1.getRoles().get(0).getRname());
        }

    }
//    @Test
//    void contextLoads() {
//
//    }

}
