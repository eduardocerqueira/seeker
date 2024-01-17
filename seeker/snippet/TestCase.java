//date: 2024-01-17T16:46:48Z
//url: https://api.github.com/gists/fc02a48318e2a630287bb404bee16175
//owner: https://api.github.com/users/AlenaVainilovich

package dto;

import lombok.Builder;
import lombok.Data;

@Builder
@Data
public class TestCase {
    String title;
    String status;
    String description;
    String severity;
    String priority;
    String type;
    String layer;
    String isFlaky;
    String behavior;
    String automationStatus;
    String preConditions;
    String postConditions;

}
