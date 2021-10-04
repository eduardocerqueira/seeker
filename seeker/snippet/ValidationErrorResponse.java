//date: 2021-10-04T16:59:56Z
//url: https://api.github.com/gists/f45c7b22c1ebbf0b5259ee6efc39374e
//owner: https://api.github.com/users/CodeVaDOs

package ua.kiev.kmrf.scheduler.dto.response.error;

import lombok.AllArgsConstructor;
import lombok.Data;
import ua.kiev.kmrf.scheduler.util.Violation;

import java.util.List;

@Data
@AllArgsConstructor
public class ValidationErrorResponse {
    private List<Violation> violations;
}