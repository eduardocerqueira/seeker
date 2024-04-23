//date: 2024-04-23T16:54:41Z
//url: https://api.github.com/gists/d67c685a508976924ac7df2a51dbcc0e
//owner: https://api.github.com/users/dynac01

package models;

import lombok.Data;

import java.util.List;

@Data
public class FedoraJsonResult<T> {
    private List<T> results;
}
