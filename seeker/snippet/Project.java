//date: 2024-01-17T16:46:48Z
//url: https://api.github.com/gists/fc02a48318e2a630287bb404bee16175
//owner: https://api.github.com/users/AlenaVainilovich

package dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class Project {
    String projectName;
    String projectCode;
    String description;
}
