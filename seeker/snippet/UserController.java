//date: 2023-02-07T17:05:22Z
//url: https://api.github.com/gists/7ac936bd057776f5a8890d8703a0e77c
//owner: https://api.github.com/users/AronKeener


package com.itheima.cntroller;

import com.itheima.been.User;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

/*
    关于"@Component"
    用于自动检测和使用类路径扫描自动配置bean。注释类和bean之间存在隐式的一对一映射（即每个类一个bean）。
    这种方法对需要进行逻辑处理的控制非常有限，因为它纯粹是声明性的
*/
@Controller
public class UserController {

    /*
        名称：@RequestMapping
        类型：方法注解
        位置：SpringMVC控制器方法定义上方
        作用：设置当前控制器方法请求访问路径
        属性：value(默认)：请求访问路径;
             method：http请求动作，标准动作（GET/POST/PUT/DELETE)
    */

    /*
        名称：@PathVariable
        类型：形参注解
        位置：SpringMVC控制器方法形参定义前面
        作用：绑定路径参数与处理器方法形参间的关系，要求路径参数名与形参名一一对应
     */

    /*
        @RequestBody，@RequestParam，@PathVariable
        区别： @RequestParam 用于接收 url 地址传参或表单传参
              @RequestBody  用于接收 json 数据
              @PathVariable 用于接收路径参数，使用{参数名称}描述路径参数
        应用： 后期开发中，发送请求参数超过 1 个时，以json格式为主，@RequestBody 应用较广
              如果发送非 json 格式数据，选用 @RequestParam 接收请求参数
              采用 RESTful 进行开发，当参数数量较少时，例如 1 个，可以采用 @PathVariable接收请求路径变量，通常用于传递id值
    */

    // 添加新用户
    @RequestMapping(value = "/users",method = RequestMethod.POST)
    @ResponseBody
    public String save(){
        System.out.println("user save ... ");
        return "{'module':'user save'}";
    }

    // 删除用户
    @RequestMapping(value = "/users/{id}",method = RequestMethod.DELETE)
    @ResponseBody
    public String delete(@PathVariable Integer id){
        System.out.println("user delete..." + id);
        return "{'module':'user delete'}";
    }

    // 更新用户
    @RequestMapping(value = "/users",method = RequestMethod.PUT)
    @ResponseBody
    public String update(@RequestBody User user){
        System.out.println("user update..." + user);
        return "{'module':'user update'}";
    }

    // 查询指定用户
    @RequestMapping("/user/{id}")
    @ResponseBody
    public String getById(Integer id){
        System.out.println("user getById ..." + id);
        return "'{module}' : user getById";
    }

    // 查询全部用户
    @RequestMapping(value = "/users",method = RequestMethod.GET)
    @ResponseBody
    public String getAll(){
        System.out.println("user getAll");
        return "user getAll";
    }
}
