//date: 2022-03-16T17:03:49Z
//url: https://api.github.com/gists/19b5d41d46990487ef23ddf8c5ca0715
//owner: https://api.github.com/users/Carlsir

package com.Zhujiaming.week3.demo;

import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.*;
import java.io.IOException;

@WebServlet(name = "LifeCycleServlet", value = "/LifeCycleServlet")
public class LifeCycleServlet extends HttpServlet {
    public LifeCycleServlet(){
        System.out.println("I am in constructor -->LifeCycleServlet()");
    }
    @Override
    public void init(){
        System.out.println("I am in init()");
    }
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        System.out.println("I am in server()-->doGet()");
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {

    }
    @Override
    public void destroy(){
        System.out.println("I am in destroy()");
    }
}