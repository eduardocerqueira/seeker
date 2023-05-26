//date: 2023-05-26T16:49:07Z
//url: https://api.github.com/gists/728d342e1ed8d753f1924d481bfb3b2a
//owner: https://api.github.com/users/tomatophobia

import com.linecorp.armeria.common.HttpResponse;
import com.linecorp.armeria.common.HttpStatus;
import com.linecorp.armeria.server.annotation.*;
import more.practice.armeriaspring.model.Todo;
import more.practice.armeriaspring.service.TodoService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;

import java.util.List;

@Controller
@PathPrefix("/todos")
public class TodoAnnotatedService {

    private final TodoService todoService;

    @Autowired
    public TodoAnnotatedService(TodoService todoService) {
        this.todoService = todoService;
    }

    @Get("/:id")
    public HttpResponse get(@Param Integer id) {
        Todo todo = todoService.get(id);
        if (todo == null) {
            HttpResponse.of(HttpStatus.NO_CONTENT);
        }
        return HttpResponse.ofJson(todo);
    }

    @Post
    public HttpResponse create(Todo todo) {
        final int result = todoService.create(todo);
        if (result == 0) {
            return HttpResponse.of(HttpStatus.INTERNAL_SERVER_ERROR);
        }
        return HttpResponse.of(HttpStatus.CREATED);
    }
}